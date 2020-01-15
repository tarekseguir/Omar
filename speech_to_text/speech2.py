import speech_recognition as sr
import time
#tab=list()
f=open("outputs.txt","w") 

def callback(recognizer, audio):                          # this is called from the background thread
    try:
        #r.adjust_for_ambient_noise(audio,duration=0.5)
        #recognizer.adjust_for_ambient_noise(audio, duration = 1)
        recognizer.dynamic_energy_threshold = True
##        recognizer_instance.dynamic_energy_adjustment_damping = 0.15
##        This value should be between 0 and 1 When this value is 1, dynamic adjustment has no effect
##        Lower values allow for faster adjustment, but also make it more likely to miss certain phrases (especially those with slowly changing volume)
        recognizer_instance.dynamic_energy_adjustment_ratio = 1.5
        recognizer_instance.adjust_for_ambient_noise(source, duration = 1)
##        the default value of 1.5 means that speech is at least 1.5 times louder than ambient noise.
        recognizer.pause_threshold = 1 #Represents the minimum length of silence (in seconds) that will register as the end of a phrase. Can be changed.
        w=recognizer.recognize_google(audio,language = "en-US")
        with open("outputs.txt","a") as f:
            f.write(w+'\n')
        #tab.append(w)
        print("You said " + w)  # received audio data, now need to recognize it
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    except LookupError:
        print("Oops! Didn't catch that")
    except Exception as e:
            #callback(recognizer,audio)
            print("")
#Creating a Recognizer instance
r = sr.Recognizer()
stop_listening =r.listen_in_background(sr.Microphone(), callback)#r:recognizer #sr.Microphone:audio
while 1:
    time.sleep(0.1)
##stop_listening(wait_for_stop=False)
##stop_listening()
