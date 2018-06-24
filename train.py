import tensorflow as tf
from dec.dataset import *
import os
import configargparse
from dec.model import *
import numpy as np

def train(dataset, \
          batch_size=256, \
          encoder_dims=[500, 500, 2000, 10], \
          initialize_iteration=50000, \
          finetune_iteration=100000, \
          pretrained_ae_ckpt_path=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    if dataset=='MNIST':
        data = MNIST()
    else:
        assert False, "Undefined dataset."

    model = DEC(params={
        "encoder_dims": encoder_dims,
        "n_clusters": data.num_classes,
        "input_dim": data.feature_dim,
        "alpha": 1.0
    })
    
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
    
    # phase 1: parameter initialization
    log_interval = 5000
    if pretrained_ae_ckpt_path==None:
        sae = StackedAutoEncoder(encoder_dims=encoder_dims, input_dim=data.feature_dim) # graph 분리?
        ae_ckpt_path = os.path.join('ae_ckpt', 'model.ckpt')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            # initialize sae
            next_ = data.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)
            cur_ae_data = data.train_x
            for i, sub_ae in enumerate(sae.layerwise_autoencoders):
                # train sub_ae
                for iter_, (batch_x, _, _) in enumerate(next_):
                    _, loss = sess.run([sub_ae.optimizer, sub_ae.loss], feed_dict={sub_ae.input_: batch_x, \
                                                                                   sub_ae.keep_prob: 0.8})
                    if iter_%log_interval==0:
                        print("[SAE-{}] iter: {}\tloss: {}".format(i, iter_, loss))
                        
                # assign pretrained sub_ae's weight
                encoder_w_assign_op, encoder_b_assign_op = model.ae.layers[i].get_assign_ops( sub_ae.layers[0] )
                decoder_w_assign_op, decoder_b_assign_op = model.ae.layers[(i+1)*-1].get_assign_ops( sub_ae.layers[1] )
                _ = sess.run([encoder_w_assign_op, encoder_b_assign_op, \
                              decoder_w_assign_op, decoder_b_assign_op])

                # get next sub_ae's input
                cur_ae_data = sess.run(sub_ae.encoder, feed_dict={sub_ae.input_: cur_ae_data, \
                                                                   sub_ae.keep_prob: 1.0})
                embedding = Dataset(train_x=cur_ae_data, train_y=cur_ae_data)
                next_ = embedding.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)
                
            # finetune AE
            for iter_, (batch_x, _, _) in enumerate(data.gen_next_batch(batch_size=batch_size, is_train_set=True, \
                                                                        iteration=finetune_iteration)):
                _, loss = sess.run([model.ae.optimizer, model.ae.loss], feed_dict={model.ae.input_: batch_x, \
                                                                                   model.ae.keep_prob: 1.0})
                if iter_%log_interval==0:
                    print("[AE-finetune] iter: {}\tloss: {}".format(iter_, loss))
            saver.save(sess, ae_ckpt_path)

    else:
        ae_ckpt_path = pretrained_ae_ckpt_path
        
        
    # phase 2: parameter optimization
    dec_ckpt_path = os.path.join('dec_ckpt', 'model.ckpt')
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ae_ckpt_path)
        
        # initialize mu
        z=sess.run(model.ae.encoder, feed_dict={model.ae.input_: data.train_x, model.ae.keep_prob: 1.0})
        assign_mu_op = model.get_assign_cluster_centers_op(z)
        _ = sess.run(assign_mu_op)

        for cur_epoch in range(50):
            q = sess.run(model.q, feed_dict={model.ae.input_: data.train_x, \
                                            model.ae.input_batch_size: data.train_x.shape[0], \
                                            model.ae.keep_prob: 1.0})
            p = model.target_distribution(q)
            
            # per one epoch
            for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size, \
                                                                                       is_train_set=True, epoch=1)):
                batch_p = p[batch_idxs]
                _, loss, pred = sess.run([model.optimizer, model.loss, model.pred], \
                                   feed_dict={model.ae.input_: batch_x, \
                                            model.ae.input_batch_size: batch_x.shape[0], \
                                            model.p: batch_p, \
                                            model.ae.keep_prob: 0.8})
            print("[DEC] epoch: {}\tloss: {}\tacc: {}".format(cur_epoch, loss, model.cluster_acc(batch_y, pred)))
            saver.save(sess, dec_ckpt_path)
     
    
    
if __name__=="__main__":
    parser = configargparse.ArgParser()
    parser.add("--batch-size", dest="batch_size", help="Train Batch Size", default=300, type=int)
    parser.add("--gpu-index", dest="gpu_index", help="GPU Index Number", default="0", type=str)

    args = vars(parser.parse_args())
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_index']

    train(batch_size=args['batch_size'],
          dataset="MNIST",
          pretrained_ae_ckpt_path="./ae_ckpt/model.ckpt")
#           pretrained_ae_ckpt_path=None)