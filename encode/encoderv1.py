import time
from picozero import pico_led, LED
from servo import Servo

encode_dict = {
    "a":"00000",
    "b":"00001",
    "c":"00010",
    "d":"00011",
    "e":"00100",
    "f":"00101",
    "g":"00110",
    "h":"00111",
    "i":"01000",
    "j":"01001",
    "k":"01010",
    "l":"01011",
    "m":"01100",
    "n":"01101",
    "o":"01111",
    "p":"10000",
    "q":"10001",
    "r":"10010",
    "s":"10011",
    "t":"10100",
    "u":"10101",
    "v":"10110",
    "w":"10111",
    "x":"11000",
    "y":"11001",
    "z":"11010",
    " ":"11011",
    ",":"11100",
    ".":"11101",
    "!":"11110",
    "?":"11111"
}

arms = Servo(pin_id=0)#15
head = Servo(pin_id=14)
waist  = Servo(pin_id=15)#14
eye1 = LED(16)
eye2 = LED(5)
string = "abcdefghijklmnopqrstuvwxyz, "
wordlist = list(string)
comm = []

def armsmv(arms,bit):
    if bit == '1':
        arms.write(145)
        time.sleep(0.5)
        eye1.off()
        eye2.off()

    else:
        arms.write(85)
        time.sleep(0.5)
        eye1.off()
        eye2.off()


def headmv(head,bit):
    if bit == '1':
        head.write(50)
        time.sleep(0.5)
    else:
        head.write(90)
        time.sleep(0.5)

def eyesmv(eye1,eye2,bits):
    if bits[0] == '0' and bits[1] == '0':
        time.sleep(0.5)
    elif bits[0] == '0' and bits[1] == '1':
        eye2.on()
        time.sleep(0.5)
    elif bits[0] == '1' and bits[1] == '0':
        eye1.on()
        time.sleep(0.5)
    elif bits[0] == '1' and bits[1] == '1':
        eye1.on()
        eye2.on()
        time.sleep(0.5)
    else:
        print("Error 1")
        
def waistmv(waist,bit):
    if bit == '1':
        waist.write(0)
        time.sleep(0.5)
    else:
        waist.write(200)
        time.sleep(0.5)
        
def reset(head,arms,waist,eye1,eye2):
    head.write(90)
    time.sleep(0.1)
    arms.write(110)
    time.sleep(0.1)
    waist.write(0)
    time.sleep(0.1)
    eye1.off()
    time.sleep(0.1)
    eye2.off()
    time.sleep(2)
    
#print(wordlist[0])
for x in range(0,len(wordlist)):
    comm.append(encode_dict[wordlist[x]])
    
for i in range(0,len(comm)):
    print(wordlist[i])
    commin = list(comm[i])
    armsmv(arms,commin[3])
    headmv(head,commin[4])
    eyesmv(eye1,eye2,commin[1:3])
    waistmv(waist,commin[0])
    time.sleep(5)
    input()
    #reset(head,arms,waist,eye1,eye2)
    
    
    
