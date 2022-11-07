import sys
import time
import numpy as np
from configparser import ConfigParser
from xarm.wrapper import XArmAPI

class XArm:

	def __init__(self, config_file='./robot.conf'):
		self.arm = None
		self.home = (150.0, -10, 170)
		self.joint_limits = None
		# self.ip = None
		self.ip = '192.168.1.246'

		# try:
		# 	parser = ConfigParser()
		# 	parser.read(config_file)
		# 	self.ip = parser.get('xArm', 'ip')
		# except:
		# 	self.ip = input('Please input the xArm ip address:')
		# 	if not self.ip:
		# 		print('input error, exit')
		# 		sys.exit(1)

	def start_robot(self):
		if self.ip is None:
			raise Exception('IP not provided.')
		self.arm = XArmAPI(self.ip, is_radian=True)
		self.clear_errors()
		self.arm.motion_enable(enable=True)

	def set_mode_and_state(self, mode=0, state=0):
		self.arm.set_mode(mode)
		self.arm.set_state(state)

	def clear_errors(self):
		self.arm.clean_warn()
		self.arm.clean_error()

	def reset(self, home = False):
		if self.arm.has_err_warn:
			self.clear_errors()
		if home:
			self.move_to_home()

	def move_to_home(self):
		self.arm.set_position(x=150+self.home[0], y=79+self.home[1], z=40+self.home[2])
		self.arm.set_gripper_position(850) #350)

	def set_position(self, pos):
		x = pos[0]*100 + self.home[0]
		y = pos[1]*100 + self.home[1]
		z = pos[2]*100 + self.home[2]
		self.arm.set_position(x=x, y=y, z=z)

	def get_position(self):
		pos = self.arm.get_position()[1]
		# print(pos)
		x = (pos[0] - self.home[0])/100.0
		y = (pos[1] - self.home[1])/100.0
		z = (pos[2] - self.home[2])/100.0
		return np.array([x,y,z]).astype(np.float32)


	def get_gripper_position(self):
		code, pos = self.arm.get_gripper_position()
		if code!=0:
			raise Exception('Correct gripper angle cannot be obtained.')
		return pos

	def set_gripper_position(self, pos, wait=False):
		'''
		wait: To wait till completion of action or not
		'''
		# if pos<200:
		# 	pos = 200
		# elif pos>400:
		# 	pos = 400
		self.arm.set_gripper_position(pos, wait=wait, auto_enable=True)

	def goto_zero(self):
		self.arm.move_gohome()

