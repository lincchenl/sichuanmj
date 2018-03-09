# -*-coding:utf-8-*-
import sichuanmj as sc
import numpy as np
global root,listb,text,server
def main():
	server=sc.sichuanmj_server()
	server.restart(1)
	for i in range(4):
		server.dingque(i)
	server.input(0)

if __name__ == "__main__":
	main()