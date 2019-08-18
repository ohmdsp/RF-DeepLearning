'''This file defines a RFDataReader class, which provides functionality to load
in raw RF data and import it in a format suitable for training and testing using
machine learning algorithms.'''

# Define imports
import pickle
import numpy as np
from pprint import *
import matplotlib.pyplot as plt
import scipy.signal as signal

# Class to read in raw RF data
# Expects an input .pkl file
# Data should be a dictionary of this particular type:
# {(MOD, SNR): Nx2x128 array}
class RFDataReader:
    # Define constructor
    def __init__(self, input_file, fs):
        # Read in the data
        #data = pickle.load(open(input_file, 'rb'))
        #data = pickle.load(open(input_file, encoding='latin1', 'rb'))
        with open(input_file, 'rb') as f:
            data= pickle.load(f, encoding='latin1') 

        # Determine modulation types and SNR ranges in data
        mods = [pair[0] for pair in data.keys()]
        snrs = [pair[1] for pair in data.keys()]
        self.mods = np.unique(mods)
        self.snrs = np.unique(snrs)

        # Display this data to the user
        print('\nModulation types present in data:\n')
        [pprint(mod) for mod in self.mods]
        print('\nSNR ranges for each modulation type in data:\n')
        [pprint(snr) for snr in self.snrs]

        # Store sample rate of data
        self.fs = fs

        # Store raw data we loaded in
        self.rawdata = data

        # Initialize other attributes
        self.trainIandQ = []
        self.trainAmpPhase = []
        self.trainMods = []
        self.trainSNRs = []
        self.trainOutLabels = []

        self.testIandQ = []
        self.testAmpPhase = []
        self.testMods = []
        self.testSNRs = []
        self.testOutLabels = []

        self.snrs_used = []
        self.mods_used = []

        self.outputClasses = []
        self.labels = []

    # Handy function for Monte Carlo processing (re-initialize all data containers)
    def reinit(self):
        # Initialize other attributes
        self.trainIandQ = []
        self.trainAmpPhase = []
        self.trainMods = []
        self.trainSNRs = []
        self.trainOutLabels = []

        self.testIandQ = []
        self.testAmpPhase = []
        self.testMods = []
        self.testSNRs = []
        self.testOutLabels = []

        self.snrs_used = []
        self.mods_used = []

    # Function that returns 3D array from a MOD and SNR input
    def queryAll(self, mod, snr):
        return self.rawdata[(mod, snr)]

    # Return a complex valued snaps
    def querySnap(self, mod, snr, entry):
        localTmpData = self.rawdata[(mod, snr)][entry]
        return localTmpData

    # Function that plots a time snapshot of an entry of a (MOD, SNR) pair
    def snapPlot(self, mod, snr, entry, normalize=False):
        # Create our time vector
        time = np.linspace(0, 128 / self.fs, 128)

        # Get the 2x128 entry that we care about
        localData = self.rawdata[(mod, snr)][entry]
        localDataComplex = [complex(data[0], data[1]) for data in localData.transpose()]

        # Get the magnitude of this data (sqrt(I^2 + Q^2))
        localMagData = np.abs(localDataComplex)

        # Normalize the data if desired
        if normalize:
            maxVal = max(localMagData)
            localDataComplex /= maxVal
            localMagData = np.abs(localDataComplex)
            localData[0] /= maxVal
            localData[1] /= maxVal

        # Get the phase information from this data
        localPhaseData = np.angle(localDataComplex)
        localPhaseData = (localPhaseData + 2*np.pi) % (2*np.pi)

        # Compute power spectrum by taking the FFT
        localPowerSpectrumData = np.abs(np.fft.fft(localDataComplex)) ** 2

        # Get the proper frequency range for this computation
        freqs = np.fft.fftfreq(len(localDataComplex), 1/self.fs)
        idxs = np.argsort(freqs)

        # Create plots to show these results for a given snapshot
        fig1 = plt.figure(figsize=(5.0, 6.0))
        plt.subplot(3, 1, 1)
        plt.title('MOD = ' + mod + ' | SNR = ' + str(snr) + ' dB' + ' | Entry = ' + str(entry))

        # Show time snapshot
        plt.plot(time, localData[0], '-r', label='I', linewidth=3)
        plt.plot(time, localData[1], '-g', label='Q', linewidth=3)
        plt.plot(time, localMagData, '-b', label='mag', linewidth=3)

        plt.legend(loc='lower right')
        plt.xlabel('Time (sec)')
        plt.ylabel('Magnitude')

        # Show phase
        plt.subplot(3, 1, 2)
        plt.plot(time, localPhaseData, '-r', label='phase', linewidth=3)
        plt.title('Phase of Specified Snapshot')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (rad)')

        # Show power spectrum
        plt.subplot(3, 1, 3)
        plt.plot(freqs[idxs], localPowerSpectrumData[idxs], '-k', linewidth=3)
        plt.title('Power Spectrum of Specified Snapshot')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')

        plt.tight_layout()
        #plt.draw()
        plt.show()

     

        # Show amplitude/phase data as a 128x2 image
        ampPhaseIm = np.row_stack((localMagData, localPhaseData))

        fig2 = plt.figure(figsize=(4.0, 4.0))
        ax2 = fig2.add_subplot(111)
        ax2.set_aspect(1)
        res3 = ax2.imshow(ampPhaseIm, cmap=plt.cm.jet, interpolation='nearest', aspect='auto')
        fig2.colorbar(res3)

        plt.xlabel('Time (SA)')
        plt.title('Amplitude/Phase 128x2 Input Data Stream')
        plt.yticks(np.arange(2), ['Amp', 'Phase'])

        plt.tight_layout()
        plt.draw()

        # Show I/Q data as a 128x2 image
        IQIm = np.row_stack((localData[0], localData[1]))

        fig3 = plt.figure(figsize=(4.0, 4.0))
        ax3 = fig3.add_subplot(111)
        ax3.set_aspect(1)
        res3 = ax3.imshow(IQIm, cmap=plt.cm.jet, interpolation='nearest', aspect='auto')
        fig3.colorbar(res3)

        plt.xlabel('Time (SA)')
        plt.title('In Phase/Quadrature 128x2 Input Data Stream')
        plt.yticks(np.arange(2), ['I', 'Q'])

        plt.tight_layout()
        plt.draw()

        # Plot spectrogram
        # Looks like total garbage for these short time bursty signals, keep it in for later
        '''
        fig4 = plt.figure(figsize=(4.0, 4.0))
        f, t, Sxx = signal.spectrogram(localDataComplex, self.fs)
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')

        plt.tight_layout()
        plt.draw()
        '''

    # Separates data into training and test data sets
    # Uses a training ratio to compute number of entries to use for training and testing
    # This function also formats the data so it's easy to use in TensorFlow
    def create_train_and_test_sets(self, trainingRatio=0.8, snrs_used=[],
                                   mods_used=[], outputClasses=[],
                                   labels=[], normalize=False):
        print('Creating training and testing data sets...\n')
        initFlag = True

        # If nothing is specified for SNR and Modulation ranges, use everything we have by default
        # If it is specified, then just use that
        if not snrs_used:
            self.snrs_used = self.snrs
        else:
            self.snrs_used = snrs_used

        if not mods_used:
            self.mods_used = self.mods
        else:
            self.mods_used = mods_used

        # If nothing is specified for the desired output classes and corresponding labels for each modulation, then
        # assume that every modulation is it's own output class and label accordingly
        if not outputClasses:
            self.outputClasses = mods_used
        else:
            self.outputClasses = outputClasses

        if not labels:
            self.labels = range(0, len(outputClasses)-1)
        else:
            self.labels = labels

        for mod_idx, mod in enumerate(self.mods_used):
            numTrain = 0
            numTest = 0
            print('Modulation: ', mod)
            for snr_idx, snr in enumerate(self.snrs_used):
                # Query raw data for this modulation and SNR pair
                dataIQ = self.queryAll(mod, snr)
                dataAmpPhase = np.zeros(dataIQ.shape)

                # Normalize the data if desired
                # We will normalize by magnitude so that all data in magnitude space is between 0 and 1
                if normalize:
                    # Go through every data record manually
                    for data_element in dataIQ:
                        # Compute magnitude data
                        localDataComplex = [complex(data_entry[0], data_entry[1]) for data_entry in data_element.transpose()]
                        localMagData = np.abs(localDataComplex)
                        maxVal = max(localMagData)

                        # Normalize I and Q samples by max magnitude
                        data_element /= maxVal

                # Also keep track of Amp/Phase data long with the raw I/Q data
                ctr = 0
                for data_element in dataIQ:
                    # Compute magnitude and phase
                    localDataComplex = [complex(data_entry[0], data_entry[1]) for data_entry in data_element.transpose()]
                    localMagData = np.abs(localDataComplex)
                    localPhaseData = np.angle(localDataComplex)

                    # Make sure phase data is not negative, between 0 and 2*pi
                    localPhaseData = (localPhaseData + 2 * np.pi) % (2 * np.pi)

                    localAmpPhase = np.zeros((2, 128))
                    localAmpPhase[0] = localMagData
                    localAmpPhase[1] = localPhaseData

                    dataAmpPhase[ctr] = localAmpPhase
                    ctr += 1

                # Compute number of entries we have, and create random permutation of indices
                randomInds = np.random.permutation(range(dataIQ.shape[0]))

                # Separate indices based on our training ratio
                trainExamples = int(trainingRatio * len(randomInds))

                # Grab our local data snapshot for training and testing
                localTrainDataIQ = dataIQ[:trainExamples]
                localTrainDataAmpPhase = dataAmpPhase[:trainExamples]
                localTestDataIQ = dataIQ[trainExamples:]
                localTestDataAmpPhase = dataAmpPhase[trainExamples:]

                # Update our counters for number of elements
                numTrain += len(localTrainDataIQ)
                numTest += len(localTestDataIQ)

                # Create the training set
                if initFlag:
                    trainDataIQ = localTrainDataIQ
                    trainDataAmpPhase = localTrainDataAmpPhase
                    trainModLabels = np.ones(trainExamples) * np.where(np.array(self.mods_used) == mod)[0]
                    trainSNRLabels = np.ones(trainExamples) * np.where(np.array(self.snrs_used) == snr)[0]
                    trainOutLabels = np.ones(trainExamples) * self.labels[mod_idx]
                else:
                    trainDataIQ = np.vstack((trainDataIQ, dataIQ[:trainExamples]))
                    trainDataAmpPhase = np.vstack((trainDataAmpPhase, dataAmpPhase[:trainExamples]))
                    trainModLabels = np.append(trainModLabels, np.ones(trainExamples) * np.where(np.array(self.mods_used) == mod)[0])
                    trainSNRLabels = np.append(trainSNRLabels, np.ones(trainExamples) * np.where(np.array(self.snrs_used) == snr)[0])
                    trainOutLabels = np.append(trainOutLabels, np.ones(trainExamples) * self.labels[mod_idx])

                # Create the test set
                if initFlag:
                    testDataIQ = localTestDataIQ
                    testDataAmpPhase = localTestDataAmpPhase
                    testModLabels = np.ones(dataIQ.shape[0] - trainExamples) * np.where(np.array(self.mods_used) == mod)[0]
                    testSNRLabels = np.ones(dataIQ.shape[0] - trainExamples) * np.where(np.array(self.snrs_used) == snr)[0]
                    testOutLabels = np.ones(dataIQ.shape[0] - trainExamples) * self.labels[mod_idx]
                    initFlag = False
                else:
                    testDataIQ = np.vstack((testDataIQ, dataIQ[trainExamples:]))
                    testDataAmpPhase = np.vstack((testDataAmpPhase, dataAmpPhase[trainExamples:]))
                    testModLabels = np.append(testModLabels,
                                              np.ones(dataIQ.shape[0] - trainExamples) * np.where(np.array(self.mods_used) == mod)[0])
                    testSNRLabels = np.append(testSNRLabels,
                                              np.ones(dataIQ.shape[0] - trainExamples) * np.where(np.array(self.snrs_used) == snr)[0])
                    testOutLabels = np.append(testOutLabels,
                                              np.ones(dataIQ.shape[0] - trainExamples) * self.labels[mod_idx])

            print('Number of training samples: ', numTrain)
            print('Number of testing samples: ', numTest)
            print('')

        # Error checking on our data shaping process
        assert(len(trainSNRLabels) == len(trainModLabels) == len(trainOutLabels))
        assert(len(testSNRLabels) == len(testModLabels) == len(testOutLabels))

        # Randomly reorder our data before we're done
        trainInds = np.random.permutation(range(len(trainModLabels)))
        self.trainIandQ = trainDataIQ[trainInds]
        self.trainAmpPhase = trainDataAmpPhase[trainInds]
        self.trainMods = trainModLabels[trainInds]
        self.trainSNRs = trainSNRLabels[trainInds]
        self.trainOutLabels = trainOutLabels[trainInds]

        testInds = np.random.permutation(range(len(testModLabels)))
        self.testIandQ = testDataIQ[testInds]
        self.testAmpPhase = testDataAmpPhase[testInds]
        self.testMods = testModLabels[testInds]
        self.testSNRs = testSNRLabels[testInds]
        self.testOutLabels = testOutLabels[testInds]

