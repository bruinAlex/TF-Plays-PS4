## Citation: Box Of Hats (https://github.com/Box-Of-Hats )
# Get pywin32 from https://www.lfd.uci.edu/~gohlke/pythonlibs/

import win32api as wapi
import time

def mouse_check():
	mouse_buttons = 0
	# Left button down = 0 or 1. Button up = -127 or -128
	if wapi.GetAsyncKeyState(0x01):
		mouse_buttons = 1
	return mouse_buttons
