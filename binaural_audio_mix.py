import glob
import numpy as np
import soundfile as sf
import sofa
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from IPython.display import Audio
from pprint import pprint
from typing import Union, Tuple, List

# find the nearest angle within the sofa dataset to a target angle
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
    
def fs_resample(s1,f1,s2,f2):
    if f1!=f2:
        if f2<f1:
            s2 = librosa.core.resample(y=s2.transpose(), orig_sr=f2, target_sr=f1)
            s2 = s2.transpose()
        else:
            s1 = librosa.core.resample(y=s1.transpose(), orig_sr=f1, target_sr=f2)
            s1 = s1.transpose()
        fmax = max([f1,f2])
        f1 = fmax
        f2 = fmax
        print('Resampled at: ', fmax, 'Hz')
        return s1, f1, s2, f2
    
def randomDisplay(N: int = 1, set_index: int = 0, target_fs: int = 48000, repetition: bool = False) -> Union[np.ndarray, int]:
    # N: number of sources/angles we want to spatialize at random angles
    # set_index: desired HRTF dataset
    # target_fs: desired sample rate
    # repetition: if True, can re-use some source samples
    # Returns:
    #  Stereo3D: Binauralized audio data (2-channels) for headphones
    
    if N<1 or N>18:
        print("Error: N needs to be between 1 and 18!" )
        return -1
    
    # Init
    HRTF = sofa.Database.open(_SOFA[set_index])
    fs_H = HRTF.Data.SamplingRate.get_values()[0]
    positions = HRTF.Source.Position.get_values(system='spherical')
    angles = np.arange(0, 360, 10)
    elevations = [-45, 0, 45]
    sources = np.arange(0,len(_SOURCES))
    H = np.zeros([HRTF.Dimensions.N,2])
    Stereo3D = np.zeros([HRTF.Dimensions.N,2])
    
    print('Using HRTF set: ' + _SOFA[set_index])
    try:
        print('Source distance is:', positions[0,2], ' meters \n')
    except Exception as e:
        print('No distance information available')
    
    # possible choices of angles in the horizontal plane (possible azimuth)    
    for n in range(N):
        # Random direction on horizontal plane
        angle_idx = np.random.randint(0, len(angles))
        angle = angles[angle_idx]
        angles = np.delete(angles, angle_idx) # for avoiding 2 or more sources being spatialized at the same azimuth
        # Remove angle and front/back complementary from list for next iteration 
        # for example 20 degrees on the right and 160 degrees on the right
        if angle <= 180: # reference point 90 deg
            arc_dist = 90 - angle
            complementary = 90 + arc_dist
        else: # reference point 270 deg
            arc_dist = 270 - angle
            complementary = 270 + arc_dist
        
        if arc_dist != 0:
            angles = np.delete(angles, np.where(angles == complementary))
        
        # Database specific format adjustments
        angle_label = angle
        angle = 360 - angle
        if angle == 360:
            angle = 0
        
        # Randomize elevation
        elev_idx = np.random.randint(0, len(elevations))
        elev = elevations[elev_idx] # bottom mid or high
        
        # Retrieve HRTF relative position closest to that angle
        # positions[:,0] -> all the rows of the first column that correspond to the azimuth angles of hrirs available
        # positions[:,1] -> all the rows of the second column that correspond to the elevation
        [az, az_idx] = find_nearest(positions[:,0], angle) 
        # There will be more than one position equally distant to azimuth
        # SOFA datasets contains many angles of the same azimuth but all at different elevations
        # subpositions will contain all the positions at the azimuth level that are the nearest
        subpositions = positions[np.where(positions[:,0] == az)]
        # withing the subset subpositions we can find the HRIR at the right elevation el
        [el, sub_idx] = find_nearest(subpositions[:,1], elev)
        # sub_idx and az_idx are the indices that will point us to the HRTF closest to our target values
        H[:,0] = HRTF.Data.IR.get_values(indices = {"M":az_idx + sub_idx, "R":0, "E":0}) # left ear of the HRTF 'E'=0 (emmitter)
        H[:,1] = HRTF.Data.IR.get_values(indices = {"M":az_idx + sub_idx, "R":1, "E":0}) # right ear of the HRTF 

        if fs_H != target_fs:
            H = librosa.core.resample(H.transpose(), fs_H, target_fs).transpose()
            
        # Pick random source(s)
        source_id = np.random.choice(sources)
        [x, fs_x] = sf.read(_SOURCES[source_id])
        try:
            if x.shape[1] > 1:
                x = np.mean(x, axis=1)
        except Exception as e:
            pass
        if not repetition:
            sources = np.delete(sources, np.where(sources==source_id))
        if fs_x != target_fs:
            x = librosa.core.resample(y=x.transpose(), orig_sr=fs_x, target_sr=target_fs).transpose()
        
        # Convolve and add L R signals and create L R specialized version of the source
        rend_L = signal.fftconvolve(x, H[:,0])
        rend_R = signal.fftconvolve(x, H[:,1])
        # helps to later normalize and scale the amplitude of the rendering to peak at 1
        # for avoiding clipping not having the signal go beyond +- 1
        # for scaling the amplitude of each source to have almost equal loudness among sources in the sound field
        M = np.max([np.abs(rend_L), np.abs(rend_R)]) 
        
        # place rendered versions into Stereo3D
        if len(Stereo3D) < len(rend_L):
            diff = len(rend_L) - len(Stereo3D)
            Stereo3D = np.append(Stereo3D, np.zeros([diff,2]), 0)
        Stereo3D[0:len(rend_L), 0] += (rend_L / M)
        Stereo3D[0:len(rend_R), 1] += (rend_R / M)
        
        # Print op
        print('Source #'+ str(n+1) + '("' + _SOURCES[source_id][12:] + '")' + ' rendered at azimuth: ' + str(angle_label) + ' and elevation ' +str(elev))
    
    return Stereo3D

def load_file_paths(
    source_dir: str, 
    hrtf_dir_MIT: str, 
    hrtf_dir_LISTEN: str, 
    hrtf_dir_SOFA: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Loads and returns the list of source and HRTF files from the specified directories.
    
    Args:
        source_dir (str): Directory pattern for the source .wav files.
        hrtf_dir_MIT (str): Directory pattern for the MIT HRTF .wav files.
        hrtf_dir_LISTEN (str): Directory pattern for the LISTEN HRTF .wav files.
        hrtf_dir_SOFA (str): Directory pattern for the SOFA .sofa files.
        
    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: A tuple containing lists of source files, MIT HRTF files, 
        LISTEN HRTF files, and SOFA HRTF files.
    """
    
    # Load source and HRTF files
    sources = glob.glob(source_dir)
    MIT_files = glob.glob(hrtf_dir_MIT)
    LISTEN_files = glob.glob(hrtf_dir_LISTEN)
    SOFA_files = glob.glob(hrtf_dir_SOFA)

    # Sort the lists
    MIT_files.sort()
    LISTEN_files.sort()
    SOFA_files.sort()

    # Return the file lists
    return sources, MIT_files, LISTEN_files, SOFA_files

if __name__ == '__main__':
    source_dir = 'your_directory_of_sounds'
    hrtf_dir_MIT = '.\\MIT - KEMAR\\elev0\\*.wav'
    hrtf_dir_LISTEN = '.\\LISTEN - IRCAM\\COMPENSATED\\WAV\\IRC_1002_C\\*.wav'
    hrtf_dir_SOFA = '.\\SOFA Far-Field\\*.sofa'

    _SOURCES, _MIT, _LISTEN, _SOFA = load_file_paths(source_dir, hrtf_dir_MIT, hrtf_dir_LISTEN, hrtf_dir_SOFA)

    # You can now print or use these lists as needed
    ans = input('Do you want to print the datasets? Press \'y\' if yes\nAns: ')
    if ans == 'y':
        print('Source files:')
        pprint(_SOURCES)
        print('\nMIT HRTF files:')
        pprint(_MIT)
        print('\nLISTEN HRTF files:')
        pprint(_LISTEN)
        print('\nSOFA HRTF files:')
        pprint(_SOFA)
        

    """

    Visualize one source with MIT set & use time-domain convolution

    """

    # We expect stereo data dimensions to be either 2xN or Nx2

    az_index = 32
    print('Using HRTF: ' + _MIT[az_index])
    [HRIR, fs_H] = sf.read(_MIT[az_index])
    print('Sample rate = ' +str(fs_H))
    print('Data dimensions: ', HRIR.shape)

    # Time domain visualization: Plot HRIR
    plt.plot(HRIR[:,0])
    plt.plot(HRIR[:,1])
    plt.xlabel('Time in Samples')
    plt.ylabel('Amplitude')
    plt.title('HRIR at angle: '+ _MIT[az_index][-11:]) # 125 degrees clockwise is on the rear right side
    plt.legend(['Left', 'Right'])
    plt.show()

    # Frequiency domain visualization
    nfft = len(HRIR)*8
    HRTF = np.fft.fft(HRIR, n=nfft, axis=0)
    HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1,:])
    HRTF_mag_dB = 20*np.log10(HRTF_mag)

    f_axis = np.linspace(0,fs_H/2, len(HRTF_mag_dB))
    plt.semilogx(f_axis, HRTF_mag_dB)
    plt.title('HRTF at angle: '+ _MIT[az_index][-11:])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Left', 'Right'])
    plt.show()

    # Load an audio source
    source_idx = 2
    print('Source is: ' +_SOURCES[source_idx])
    [sig, fs_s] = sf.read(_SOURCES[source_idx])
    print('Sample rate: ', fs_s)
    print('Data dimensions:', sig.shape)

    # Turn a stereo source into mono if needed
    if len(sig.shape) > 1 and sig.shape[1] > 1:
        sig_mono = np.mean(sig,axis=1)
    else:
        sig_mono = sig

    print('New data dimensions: ', sig_mono.shape)
    
    # Sample rates are different so resample to match
    [sig_mono, fs_s, HRIR, fs_H] = fs_resample(sig_mono, fs_s, HRIR, fs_H)
    print('sig dimensions: ', sig_mono.shape)
    print('hrir dimensions: ', HRIR.shape)
    
    # Time domain convolution between source signal and HRIR channels
    s_L = np.convolve(sig_mono, HRIR[:,0])
    s_R = np.convolve(sig_mono, HRIR[:,1])
    
    # Put L/R convolution results into N-rows / 2-colums matrix for stereo reproduction
    Bin_Mix = np.vstack([s_L, s_R]).transpose()
    print('Data Dimensions: ', Bin_Mix.shape) 
    
    sf.write('Example1.wav', Bin_Mix, fs_s)
     
    """

    Visualize one source with MIT set & use frequency-domain convolution (faster)

    """
    # source = '.\\a_stereo_synth.wav'
    src_0, fs_s0 = librosa.load(_SOURCES[2], mono=True, sr=48000)
    src_1, fs_s1 = librosa.load(_SOURCES[18], mono=True, sr=48000)
    
    print(_SOURCES[5])
    print(_SOURCES[18])
    
    # pprint(_LISTEN) # anti-clockwise convention
    
    # Choose HRTF
    idx = [17,125] # 1 channel front left and 1 channel back right
    
    # idx = np.random.randint(0, len(_LISTEN), 2) # R , T angle counter-clockwise
    
    # Load HRTF (use librosa for variety) - librosa imports with opposite dimensionality + allows resample immediately
    HRIR_0, fs_H0 = librosa.load(_LISTEN[idx[0]], sr=48000, mono=False)
    HRIR_1, fs_H1 = librosa.load(_LISTEN[idx[1]], sr=48000, mono=False)
    
    print(_LISTEN[idx[0]])
    print('HRIR Dim: ', HRIR_0.shape)
    print('HRIR fs: ', fs_H0)
    print(_LISTEN[idx[1]])
    print('HRIR Dim: ', HRIR_1.shape)
    print('HRIR fs: ', fs_H1)    
    
    nfft = 256
    HRTF_0 = np.fft.fft(HRIR_0.T, n=nfft, axis=0)
    HRTF_1 = np.fft.fft(HRIR_1.T, n=nfft, axis=0)
    
    # Magnitude
    HRTF_0_mag = 20*np.log10((2/nfft)*np.abs(HRTF_0[0:int(len(HRTF_0)/2)+1, :]))
    HRTF_1_mag = 20*np.log10((2/nfft)*np.abs(HRTF_1[0:int(len(HRTF_1)/2)+1, :]))
    f_axis = np.linspace(0, fs_H/2, len(HRTF_0_mag))
    
    # Visualization
    
    fig = plt.figure(figsize=(8,10))
    
    # Time-Domain
    plt.subplot(2,2,1)
    plt.plot(HRIR_0.transpose())
    plt.ylabel('Amplitude')
    plt.xlabel('Time [samples]')
    plt.title(_LISTEN[idx[0]][-28:])
    plt.ylim([-.8, .8])
    plt.legend(['Left', 'Right'])
    
    plt.subplot(2,2,2)
    plt.plot(HRIR_1.transpose())
    plt.ylabel('Amplitude')
    plt.xlabel('Time [samples]')
    plt.title(_LISTEN[idx[1]][-28:])
    plt.ylim([-.8, .8])
    plt.legend(['Left', 'Right'])
    
    # Frequency Domain
    plt.subplot(2,2,3)
    plt.semilogx(f_axis, HRTF_0_mag)
    plt.ylabel('Magnitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.legend(['Left', 'Right'])
    
    plt.subplot(2,2,4)
    plt.semilogx(f_axis, HRTF_1_mag)
    plt.ylabel('Magnitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.legend(['Left', 'Right'])
    
    plt.show()
    
    # 4 convolutions in frequency domain
    s_0_L = signal.fftconvolve(src_0, HRIR_0[0,:]) # spatializerd source 0 Left
    s_0_R = signal.fftconvolve(src_0, HRIR_0[1,:]) # spatialized source 0 Right
    s_1_L = signal.fftconvolve(src_1, HRIR_1[0,:]) # spatialized source 0 Left
    s_1_R = signal.fftconvolve(src_1, HRIR_1[1,:]) # spatialized source 0 Right
    
    # other way: multiply HRTF and take then the inverse fft
    
    # Sources length
    print('Sources length before padding:')
    print(s_0_L.shape)
    print(s_1_L.shape)
    
    # Sources have different length so to add L/R together we do zero-pad to have equal lengths
    target_len = np.max([len(s_0_L), len(s_1_L)])
    pad_0 = target_len - len(s_0_L)
    pad_1 = target_len - len(s_1_L)
    
    s_0_L = np.pad(s_0_L, (0,pad_0), 'constant')
    s_0_R = np.pad(s_0_R, (0,pad_0), 'constant')
    s_1_L = np.pad(s_1_L, (0,pad_1), 'constant')
    s_1_R = np.pad(s_1_R, (0,pad_1), 'constant')
    
    print('\nSources length after padding:')
    print(s_0_L.shape)
    print(s_1_L.shape)
    
    # Add L/R signals together to create a stereo file
    L = s_0_L + s_1_L # Left headphone channel
    R = s_0_R + s_1_R # Right headphones channel
    Bin_Mix = np.vstack([L,R]).transpose()
    print('Data Dimensions: ', Bin_Mix.shape)
    
    # Normalize to avoid signals being high in amplitude
    Bin_Mix = Bin_Mix/np.max(np.abs(Bin_Mix))
    sf.write('Example2.wav', Bin_Mix, 48000)
    plt.plot(Bin_Mix)
    
    """
    SOFA (Spatially Oriented Format for Acoustics)
        Portable file format that can be read by different applications across different systems.
        They are popular nowadays to store room impulse responses or head related impulse responses (HRIR).
        The use Spherical Far-Field HRIR Compilation of the Neumann KU100
    """
    
    print('\n\n SOFA \n\n')
    pprint(_SOFA)
    
    # SOFA metadata info about a particular set choice
    sofa.Database.open(_SOFA[0]).Metadata.dump()
    
    fs = 48000
    N = 3 # Example number of sources
    Stereo3D = randomDisplay(N=N, set_index=2, target_fs=48000, repetition=False)

    # Normalize the audio
    max_val = np.max(np.abs(Stereo3D))
    if max_val > 0:
        Stereo3D /= max_val  # Normalize to range [-1, 1]

    # Save to WAV file
    output_file = 'Example3.wav'
    sf.write(output_file, Stereo3D, 48000)

    print(f'Audio saved to {output_file}')