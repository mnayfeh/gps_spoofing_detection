import pandas as pd
import numpy as np
from mavros_msgs.msg import Mavlink
from mavros import mavlink
from pymavlink import mavutil
import rospy
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from led_msgs.srv import SetLEDs
from clover.srv import SetLEDEffect
from led_msgs.msg import LEDState

warnings.filterwarnings("ignore", category=UserWarning)

rospy.init_node('flight')

link = mavutil.mavlink.MAVLink('', 255, 1)

z = pd.read_csv('Data.csv', usecols=[3, 4, 5, 11, 12, 15, 19, 26])
z = np.array(z)
scaler = StandardScaler()
scaler.fit(z)
y = []
open('output_file.txt', "w")
f = open('output_file.txt', "a")
f.write('Count' + '\t\t' + 'Prediction' + '\t\t' + 'Accuracy' + '\t\t' + 'Actual' + '\n\n')
AttackType = 2

def mavlink_cb(msg):
	mav_msg = link.decode(mavlink.convert_to_bytes(msg))
	x = str(mav_msg)
	if "GPS" in x:
		input = [mav_msg.lat, mav_msg.lon, mav_msg.alt, mav_msg.eph, mav_msg.epv, (mav_msg.vel)/100, mav_msg.cog, mav_msg.satellites_visible]
		with open("RF.pkl", 'rb') as pickle_file:
			clf = pickle.load(pickle_file)
		input = np.array(input)
		input = input.reshape(1,-1)
		input = scaler.transform(input)
		pred = clf.predict(input)
		y.append(pred)
		q = np.array(y)
		count = (q == AttackType).sum()
		accuracy = count/len(y)
		print('Class = ', pred)
		print('Number of samples = ', len(y))
		print('Detection Rate = ', accuracy)
		f.write(str(count) + '\t\t' + str(pred) + '\t\t\t' + str(accuracy) + '\t\t\t' + str(AttackType) + '\n\n')

mavlink_sub = rospy.Subscriber('mavlink/from', Mavlink, mavlink_cb)

rospy.spin()
