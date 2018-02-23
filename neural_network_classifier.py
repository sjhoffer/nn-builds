import numpy as np
import tensorflow as tf
from load_without_folds import loadData
import os
from datetime import datetime
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import random

sess = tf.InteractiveSession()

# generate data

#####################################################################################################################################################

matrices = loadData('Example_Data_Set.csv')

input_matrix,output_matrix = matrices[0],matrices[1]

#####################################################################################################################################################

start_time = time.time()
np.set_printoptions(precision=3)

#initialize parameters

learn_rate         = 0.00005  # dropout nets should use a higher learning rate
epochs             = 500000      # dropout networks call for regularization
n_input            = input_matrix.shape[1]
drop_param         = 0.5
n_hidden           = int(round((n_input+2)/2)/(1-drop_param))
std_len            = np.sqrt(2/n_input)#-0.06
n_output           = 2
log_size           = 250
IsRegularized      = 1
lambd              = 0.05
singleDonateweight = 1
secondDonateweight = 1.5
IsSoftmaxEntropy   = 1 #1 for softmax, 0 for sigmoid
IsNotWeighted      = 0
weights            = [singleDonateweight,secondDonateweight]
shuffleFlag = True
attr_iter = 60

if IsSoftmaxEntropy:
    mdl = 'Sftmx'
else: 
    mdl = 'WeightedSgmd'

#create unique key

start = datetime.now()
curtime = time.strftime("%d_%m_%Y.%H.%M.%S")
os.chdir('tflogs')
os.makedirs(curtime)
os.chdir(curtime)
this_dir = os.getcwd()

unique = time.strftime("%d_%m_%Y.%H.%M.%S") +'__' + 'mdl..' + mdl + '__' +'nhid..'+str(n_hidden)+ '__'+ 'std..'+str(std_len) +'__'+ 'lrt..'+str(learn_rate) +'__'  +'drp..'+str(drop_param) + '__' +'lbd..' + str(lambd) + '____'

test_size = np.shape(input_matrix)[0]

#####################################################################################################################################################

#set graph structures

class_adjust = tf.cast(tf.constant(weights),tf.float32)

test_length = tf.cast(tf.constant(test_size),tf.float32)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_input] ,name="x")
    y = tf.placeholder(tf.float32, [None, n_output],name="y")

with tf.name_scope('biases'):
    b0 = tf.Variable(tf.truncated_normal([n_hidden],stddev=std_len),name="b0")
    b1 = tf.Variable(tf.truncated_normal([n_output],stddev=std_len),name="b1")

with tf.name_scope('weights'):
    w0 = tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=std_len),name="w0")
    w1 = tf.Variable(tf.truncated_normal([n_hidden,n_output],stddev=std_len),name="w1")

keep_prob = tf.placeholder(tf.float32)
prec = tf.placeholder(tf.float32)
f1 = tf.placeholder(tf.float32)


def returnPredictions(x,w0,w1,b0,b1):
    '''
    returnPredictions models a two layer neural network.  The network utilizes ReLUs to 
    move from an input layer, to a hidden layer, and then finally the output layer.tf
    The function returns h, which is a tensor containing all of the label predictions.
    
    The function also applies dropout to the hidden layer to reduce overfitting.
    '''
    
    with tf.name_scope('hidden_layer'):
        z1 = tf.add(tf.matmul(x, w0), b0,name="z1")
        a2 = tf.nn.relu(z1,name="w0")
        a2_drop = tf.nn.dropout(a2, keep_prob)
    
    with tf.name_scope('output_layer'):
        z2 = tf.add(tf.matmul(a2_drop, w1), b1,name="z2")
        h = tf.nn.relu(z2,name="h")
    
    return h  

y_ = returnPredictions(x,w0,w1,b0,b1)

softmax = tf.reduce_mean(tf.nn.softmax(logits=y_))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y))

if IsRegularized:
    reg_terms = tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1)
    loss = tf.add(loss,tf.multiply(lambd,reg_terms))

model = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss) # run Adam Optimizer training step

# metrics

#####################################################################################################################################################

logits = y_
labels = y

actual_positives = tf.cast(labels[:,1], tf.bool)
actual_negatives = tf.logical_not(actual_positives)
correct_pred   = tf.cast(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)),tf.float32), tf.bool)
false_pred     = tf.logical_not(correct_pred)
    
tp   = tf.cast(tf.logical_and(correct_pred,actual_positives),tf.float32)
tn   = tf.cast(tf.logical_and(correct_pred, actual_negatives),tf.float32)
fp   = tf.cast(tf.logical_and(false_pred, actual_negatives),tf.float32)
fn   = tf.cast(tf.logical_and(false_pred, actual_positives),tf.float32)

with tf.name_scope('tp_perc'):
    tp_perc   = tf.reduce_mean(tp)

with tf.name_scope('tn_perc'):
    tn_perc   = tf.reduce_mean(tn)

with tf.name_scope('fp_perc'):
    fp_perc   = tf.reduce_mean(fp)

with tf.name_scope('fn_perc'):
    fn_perc   = tf.reduce_mean(fn)

with tf.name_scope('prec'):
    prec = tf.divide(tp_perc,tf.add(tp_perc,fp_perc))

with tf.name_scope('rec'):
    rec  = tf.divide(tp_perc,tf.add(tp_perc,fn_perc))

with tf.name_scope('f1'):
    f1   = tf.multiply(tf.cast(2,tf.float32),tf.divide(tf.multiply(prec,rec),tf.add(prec,rec)))


with tf.name_scope('Accuracy'): # set operation for tracking accuracy
    correct_prediction = tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

conf = tf.cast(tf.confusion_matrix(labels = tf.argmax(y,1), predictions = tf.argmax(y_,1),num_classes=n_output),tf.float32)
conf = tf.divide(conf,tf.cast(tf.shape(y_)[0],tf.float32))

# tensorboard setup

#####################################################################################################################################################

def variable_summaries(var,scope,isScalar):
  """
  Attach a lot of summaries to a Tensor (for TensorBoard visualization). [Example taken from Google tutorial]
  """
  with tf.name_scope(unique + scope):
    if isScalar == 0:
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        #tf.summary.histogram('histogram', var)
    else:
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)

for i in [[softmax,'softmax',1],[b0,'b0',0],[b1,'b1',0],[w0,'w0',0],[w1,'w1',0],[loss,'loss',1],[accuracy,'accuracy',1],[prec,'prec',1],[rec,'rec',1],[tp_perc,'tp_perc',1],[tn_perc,'tn_perc',1],[fp_perc,'fp_perc',1],[fn_perc,'fn_perc',1],[f1,'f1',1]]:
    variable_summaries(i[0],i[1],i[2])

merged = tf.summary.merge_all() # merge summaries

train_writer = tf.summary.FileWriter(this_dir + '/train',sess.graph) # train writer
test_writer = tf.summary.FileWriter(this_dir + '/test')

#####################################################################################################################################################
# begin training/execution

score_tracker = np.zeros((attr_iter,5))
attribute_tracker = np.zeros((attr_iter,7))

skf = StratifiedKFold(n_splits=5,shuffle=shuffleFlag)

for j in range(0,attr_iter):

    learn_rate         = random.uniform(0.00000001,0.00015)
    drop_param         = random.uniform(0.4,0.95)
    n_hidden           = random.uniform(int(round((n_input+2)/2)/(1-drop_param))*0.5,int(round((n_input+2)/2)/(1-drop_param))*1.5)
    std_len_scaler     = random.uniform(0.75,1.25)
    std_len            = np.sqrt(2/n_input) * std_len_scaler
    lambd              = random.uniform(0.0000001,0.2)
    secondDonateweight = random.uniform(1,2.5)
    weights            = [singleDonateweight,secondDonateweight]

    attribute_tracker[j,:] = [learn_rate,drop_param,n_hidden,std_len_scaler,std_len,lambd,secondDonateweight]

    for k, (train, test) in enumerate(skf.split(input_matrix, output_matrix[:,1])):
        
        x_train, y_train = input_matrix[train,:],  output_matrix[train,:]
        x_test,  y_test  = input_matrix[test,:], output_matrix[test,:]
        
        sm = SMOTE(kind='borderline1') #better performance than regular kind

        
        x_train, labels_over = sm.fit_sample(x_train,  y_train[:,1])
        
        label_mat = np.zeros([len(labels_over),2])
        label_mat[:,1] = labels_over;
        label_mat[:,0] = 1 - labels_over;
        label_mat = label_mat.astype(np.float32)
        y_train = label_mat
        
        label_mat = np.zeros([len(y_test[:,1]),2])
        label_mat[:,1] =y_test[:,1]
        label_mat[:,0] = 1 - y_test[:,1];
        label_mat = label_mat.astype(np.float32)
        y_test    = label_mat
        
        rec_max = 0;
        f1_max = 0;
        rec_max_step = 0;
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            time_el_tot = 0
            
            for step in range(0,epochs):
                
                sess.run([model,loss],feed_dict={x: x_train, y: y_train , keep_prob: drop_param })
                
                if (step % (log_size) ==0) & (step > 2000):
                    
                    time_el_tot = round((time.time() - start_time)/60,1)           
                    print(j,k,step,'| ',time_el_tot,'minutes')
                    
                    summary_train, acc = sess.run([merged, accuracy], feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
                    train_writer.add_summary(summary_train, step)
                    summary_test, acc = sess.run([merged, accuracy], feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
                    test_writer.add_summary(summary_test, step)
                    
                    rec  = tf.divide(tp_perc,tf.add(tp_perc,fn_perc))
                    f1   = tf.multiply(tf.cast(2,tf.float32),tf.divide(tf.multiply(prec,rec),tf.add(prec,rec)))
                    rec, f1 = sess.run([rec, f1],feed_dict={x: x_test, y: y_test, keep_prob: 1.0 })
                    
                    if rec > rec_max:
                        rec_max = rec
                        f1_max = f1
                        rec_max_step = step

                    elif (rec_max_step + 3500) <= (step): 
                        print(rec_max_step,rec_max,f1_max)
                        confu2 = sess.run(conf, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
                        print(confu2)
                        break
                    
                    print(step,rec,f1)
                    print(rec_max_step,rec_max,f1_max)

        score_tracker[j,k] = (rec_max+f1_max*2)/2
        average_score = np.mean(score_tracker,1)
        print('0000000000000000000000000000000000000000')
        print(score_tracker)
        print(average_score)
        print(attribute_tracker[j,:])
print('Neural network has finished learning. Printing scores...')
print('       ')
print(score_tracker)
print(average_score)
print(attribute_tracker)

