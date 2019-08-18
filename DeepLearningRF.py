'''
Top level script that executes CNN against raw RF data
'''

# Define imports
from RFDataReader import RFDataReader
from RFNN import *
import numpy as np
from matplotlib import pyplot as plt
import os

#import pydevd
#pydevd.settrace('172.17.17.109', port=8000, stdoutToServer=True, stderrToServer=True, suspend=False)

# Define and create (if necessary) top level directory for writing
topDir = 'test'
if not os.path.exists(topDir):
    os.mkdir(topDir)

# Plot training accuracy as a function of batch
def plot_train_results(batches, train_accs, filt_ratio=1, fig_str=''):
    # Filter the training data as specified
    batches_filt = batches[0::filt_ratio]
    train_accs_filt = train_accs[0::filt_ratio]

    fig = plt.figure(figsize=(6.0, 4.0))

    # Put lables on the axes
    plt.xlabel('Batch #')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy as a Function of Batch')

    # Plot the data
    plt.plot(batches_filt, train_accs_filt, linewidth=3)
    plt.draw()

    if fig_str:
        fig.savefig(os.path.join(topDir, fig_str), bbox_inches='tight')

# Plot training accuracy as a function of SNR
def plot_test_acc_snr(y_true, y_pred, y_snr, SNRs, fig_str=''):
    acc_snr = []

    for sidx, snr in enumerate(SNRs):
        # Get indices of this SNR in our test set
        idxs = [i for i, x in enumerate(y_snr) if x == sidx]

        # Grab this part of the data
        y_true_local = y_true[idxs]
        y_pred_local = y_pred[idxs]

        # Compute accuracy on this subset
        acc_local, cts_local = compute_list_acc(y_true_local, y_pred_local)

        # Store this accuracy
        acc_snr.append(acc_local)

    # Plot test set accuracy as a function of SNR
    fig = plt.figure(figsize=(6.0, 4.0))

    plt.xlabel('SNR (dB)')
    plt.ylabel('Test Set Accuracy')
    plt.title('Test Set Accuracy as a Function of SNR')

    plt.plot(SNRs, acc_snr, linewidth=3)
    plt.draw()

    if fig_str:
        fig.savefig(os.path.join(topDir, fig_str), bbox_inches='tight')

# Function that filters results based on desired SNR ranges
def filt_by_snr(y_true, y_pred, y_mods, y_snrs, SNRs, desired_snr):
    # Get index of desired SNR within total SNR list
    sidx = SNRs.index(desired_snr)

    # Get indices of this SNR in our test set
    idxs = [i for i, x in enumerate(y_snrs) if x == sidx]

    # Grab this part of the data
    y_true_snr = y_true[idxs]
    y_local_snr = y_pred[idxs]
    y_mods_snr = y_mods[idxs]

    return y_true_snr, y_local_snr, y_mods_snr

# Function to create a confusion matrix where:
# Each modulation has a count associated with all the output labels
def create_mod_conf_mtx(y_true, y_pred, y_mods, mods, outputClasses):
    cm = []
    norm_cm = []
    # Iterate over each modulation
    for mod_idx, mod in enumerate(mods):
        # Get indicies of testing samples with this modulation type
        idxs = [i for i, x in enumerate(y_mods) if x == mod_idx]

        # Grab just this test classification data chunk
        y_true_mod = y_true[idxs]
        y_pred_mod = y_pred[idxs]

        # Compute overall accuracy for this modulation
        # Compute classification accuracy for this modulation based on specified output classes
        test_acc_mod = compute_list_acc(y_true_mod, y_pred_mod)
        print('Overall accuracy for Modulation ' + mod + ' : ' + str(test_acc_mod))

        # Compute tallies for each output class for this modulation
        mod_count_vec = []
        mod_acc_vec = []
        for lbl_idx, lbl in enumerate(outputClasses):
            mod_counts = len([i for i, x in enumerate(y_pred_mod) if x == lbl_idx])
            mod_acc = mod_counts / float(len(y_pred_mod))

            mod_count_vec.append(mod_counts)
            mod_acc_vec.append(mod_acc)

        cm.append(mod_count_vec)
        norm_cm.append(mod_acc_vec)

    return np.array(cm), np.array(norm_cm)

def plot_confusion_matrix(cm, norm_cm, title='Confusion Matrix', fig_str='', mods=[], labels=[]):
    # Plot confusion matrix
    width, height = cm.shape
    fig_cm = plt.figure(figsize=(10.0, 10.0))
    ax = fig_cm.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(norm_cm, cmap=plt.cm.jet, interpolation='nearest')
    fig_cm.colorbar(res)

    # Set title and axes
    plt.title(title)

    # Add raw count labels for each category
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', fontsize=15, color='white')

    # Add modulation labels
    plt.xticks(np.arange(0, len(labels)), labels)
    plt.yticks(np.arange(0, len(mods)), mods)

    # Save off figure
    if fig_str:
        fig_cm.savefig(os.path.join(topDir, fig_str), bbox_inches='tight')

# Compute an accuracy (ratio of matching elements between two lists of the same length)
def compute_list_acc(list1, list2):
    # Error checking
    assert(len(list1) == len(list2))

    # Compute similarity
    acc = sum(list1 == list2) / float(len(list1))

    # Compute raw counts
    cts = sum(list1 == list2)

    return acc, cts

# Code to run upon command line execution
if __name__ == "__main__":
    # Define data file input
    dataFile = '2016.04C.multisnr.pkl'
    sampleRate = 1e6 # input sample rate of dataset

    # Set up matplotlib for plotting
    plt.ion()

    # Create RFDataReader object to read in and parse the data
    # This will display some statistics of the data to the console
    RFData = RFDataReader(dataFile, sampleRate)

    #RFData.snapPlot('QAM64', 18, 100, normalize=True)
    #RFData.snapPlot('PAM4', 18, 100, normalize=True)

    # Specify which SNR ranges for input dataset processing
    SNRs = range(0, 20, 2) # SNR ranges to use in data processing

    # Specify which modulations to look at for input dataset processing
    MODs = ['AM-DSB', 'AM-SSB', 'WBFM', '8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK']

    # Specify what the output labels should be
    outputClasses = MODs
    LABELs = range(0, len(MODs))

    inputDim = [128, 2]
    numOutputClasses = len(outputClasses)

    # Create training and testing data sets
    RFData.create_train_and_test_sets(trainingRatio=0.8, snrs_used=SNRs,
                                      mods_used=MODs, outputClasses=outputClasses,
                                      labels=LABELs, normalize=True)

    RFData.snapPlot('BPSK', 18, 100, normalize=True)
    #RFData.snapPlot('QPSK', 18, 100, normalize=True)
    #RFData.snapPlot('AM-DSB', 18, 100, normalize=True)

    # Initialize our RF Neural Network
    RFNet = RFNN(inputDim, numOutputClasses)

    # Set up the first layer
    RFNet.create_conv_layer(num_feature_maps=64, patch_size=[8, 2], max_pool=True,
                            ksize=[1, 4, 2, 1], pool_strides=[1, 2, 2, 1], pool_padding='SAME')

    # Set up the third layer
    RFNet.create_fc_layer(num_nodes=512, dropout='True')

    # Set up the final layer - softmax output on output classes
    RFNet.create_readout_layer()

    # Prepare our Neural Network for training and testing
    RFNet.prepare(min_fcn='cross_entropy')

    # Train our NN on the training data set we loaded in
    train_acc = RFNet.train(RFData, data_type='IandQ', batch_size=50,
                            dropout=True, keep_prob_val=0.7)

    # Plot training accuracy as a function of the batch
    batches, train_accs = RFNet.get_train_results()

    # Test our NN on the testing data set we loaded in
    test_acc = RFNet.test(RFData, data_type='IandQ')

    # Release tensorflow resources and close this session
    RFNet.release()

    # Plot training results
    plot_train_results(batches, train_accs, filt_ratio=50)

    # Generate final results on test set by plotting confusion matrices
    cm, norm_cm = RFNet.get_conf_mtxs()

    # Plot output class v. output class confusion matrix
    plot_confusion_matrix(cm, norm_cm, title='Output Class Confusion Matrix',
                          mods=outputClasses, labels=outputClasses)

    # Create a different confusion matrix - one that shows each modulation as a
    # function of the output classes
    y_true, y_pred = RFNet.get_truth_and_predicions()

    cm_mod, norm_cm_mod = create_mod_conf_mtx(y_true, y_pred, RFData.testMods, MODs, outputClasses)

    # Plot this confusion matrix also
    plot_confusion_matrix(cm_mod, norm_cm_mod, 'Modulation Confusion Matrix',
                          mods=MODs, labels=outputClasses)

    # Plot test accuracy as a function of SNR
    plot_test_acc_snr(y_true, y_pred, RFData.testSNRs, SNRs)

    # Plot confusion matrices for 0 dB and 18 dB SNR cases
    y_true_0dB, y_pred_0dB, y_mods_0dB = filt_by_snr(y_true, y_pred, RFData.testMods, RFData.testSNRs, SNRs, 0)
    y_true_18dB, y_pred_18dB, y_mods_18dB = filt_by_snr(y_true, y_pred, RFData.testMods, RFData.testSNRs, SNRs, 18)

    cm_mod_0dB, norm_cm_mod_0dB = create_mod_conf_mtx(y_true_0dB, y_pred_0dB, y_mods_0dB, MODs, outputClasses)
    cm_mod_18dB, norm_cm_mod_18dB = create_mod_conf_mtx(y_true_18dB, y_pred_18dB, y_mods_18dB, MODs, outputClasses)

    plot_confusion_matrix(cm_mod_0dB, norm_cm_mod_0dB, 'Modulation Confusion Matrix for SNR = 0dB',
                          mods=MODs, labels=outputClasses)

    plot_confusion_matrix(cm_mod_18dB, norm_cm_mod_18dB, 'Modulation Confusion Matrix for SNR = 18dB',
                          mods=MODs, labels=outputClasses)

#    plt.show(block=True)


