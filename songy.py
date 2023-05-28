''' This project is keen on introducing time signals and transforming them into frequency domain using suitable libraries.
We will make simple musical notes of piano and plot them. Then, we are going to embed the signal with some random noise.
Afterwards, we try to purify the signal
and restore the original one. '''

#importings
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
import math
import time


''' Those are the notes we are going to generate our note from 
feel free to play with them or search on the web for more frequencies'''
t = np.linspace(0, 3, 12 * 1024) # time (the axis)
left_freq = [130.18, 146.83, 164.81, 174.61, 196, 220, 246.93] # third octave
right_freq = [261.63, 293.66, 329.63, 349.23, 392, 440, 493.88]# fourth octave

n = int(input("Please enter the number of notes U want:"))

first = []
second = []
# the loop just picks up random freq. First list representing the left hand.

for i in range(n):
    x, y = np.random.randint(0, len(left_freq) - 1, 2)
    first.append(left_freq[x])
    second.append(right_freq[y])

# transforming the lists to numpy arrays.
first = np.array(first)
second = np.array(second)

''' the song funciton generates the song signal when called given the frequency parameters.It works as follows:
1- assign the i_th division of the signal to the i_th frequency obtained from the (left & right) hands with a random time
2- repeat until we reach to the n_th frequencies'''

def song(t, first, second, n):
    x = np.zeros(t.shape)
    T = t.shape[0] // n

    for i in range(n):
        arr = np.zeros(t.shape)
        rand = np.random.randint(0, T // 4)  # random empty part of the period period
        # -I*T +(I+1)*T=T
        start = i * T
        end = (i + 1) * T - rand
        arr[start:end] = 1
        x += (np.sin(np.pi * 2 * first[i] * t) + np.sin(np.pi * 2 * second[i] * t)) * arr
    return x

# this func provides a noise when called and provided with 2 frequencies
def noise_maker(t, i, j):
    return np.sin(np.pi * 2 * i * t) + np.sin(np.pi * 2 * j * t)

# this func transforms the time domain signal to a frequency domain signal
def transform(x, N):
    x_f = fft(x)
    x_f = 2 / N * np.abs(x_f[0:N // 2])
    return x_f


# Action part

'''
calling the song with time t ,
left hand frequencies(first),
right hand frequencies (second) 
and number of divisions n      '''

x=song(t,first,second,n)

N = 3 * 1024
f = np.linspace(0, 512, int(N // 2))

# frequency domain of x
x_f = transform(x, N)

# choose 2 random frequencies between (0 - 512)
fn1,fn2 = np.random.randint(0, 512, 2)

# noise generated
noise = noise_maker(t, fn1, fn2)

# adding noise to the original signal
x_n = noise + x

# transforming the noisy signal to freq domain
x_f_n = transform(x_n, N)

# the max value rounded to the next integer
mx = math.ceil(np.max(x_f))
# getting the indecies where the noisy  singal is greater than the maximum of the original signal without noise
indecies=np.where(x_f_n>mx)
indecies=list(indecies[0])

''' Here we have three scenarios :
1- when we spotted the 2 anamoly frequencies, which is a success so we do no more actions
2- when we spotted zero indices, so the two frequencies were both zero because 
            the noise function dealt with a Sin function which gives zero at t=0
3- when we spotted just on4e freq which implies one of two cases:
   a- the two frequencies were equal so we append another value equal to the first one
   b- one was spotted and the other was just zero which does no harm to the original signal 
'''
# corresponds to number (2) above
if(len(indecies)==0):
    indecies=[0,0]

# corresponds to number (3-a)
elif(len(indecies)==1):
    indecies*=2

# get the frequencies as integer values because we know they were integers
arr=list(map(int,[f[y] for y in indecies]))

candidate_noise_to_cancel=noise_maker(t,arr[0],arr[1])
testy=x_n-x-candidate_noise_to_cancel

# corresponds to number (3-b)
if len(np.where(np.abs(testy)>1e-6)[0])!=0 : arr[0]=0

# the original noise created to be subtracted back.
noise_to_cancel = noise_maker(t, arr[0],arr[1])

# plotting the original signal, the noisy signal and the purified one
x_new = x_n - noise_to_cancel
x_f_new = transform(x_new, N)

plt.subplot(3, 2, 1)
plt.plot(t, x, c='cyan')
plt.xlabel("time")
plt.ylabel('song in t dmn')
plt.subplot(3 ,2, 2)
plt.plot(f, x_f, c='orange')
plt.xlabel("freq")
plt.ylabel('song in freq dmn')


# plt.show()

plt.subplot(3, 2, 3)
plt.plot(t, x_n, c='cyan')
plt.xlabel("time")
plt.ylabel('song in t dmn with noise')

plt.subplot(3, 2, 4)
plt.plot(f, x_f_n, c='orange')
plt.xlabel("freq")
plt.ylabel('song in freq dmn with noise')

#plt.show()


plt.subplot(3, 2, 5)
plt.plot(t, x_new, c='cyan')
plt.xlabel("time")
plt.ylabel('song in t D with noise canceled')

plt.subplot(3, 2, 6)
plt.plot(f, x_f_new, c='orange')
plt.xlabel("freq")
plt.ylabel('song in freq D with noise canceled')

plt.show()

# this function generates the sound created by the song function
sd.play(x,N)
sd.wait()

# 3 sec between playing the pure and the noisy signal
time.sleep(2)

# the sound obtained from the noisy function
sd.play(x_n,N)
sd.wait()