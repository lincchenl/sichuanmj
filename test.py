# -*-coding:utf-8-*-
import sichuanmj as sc
import numpy as np
global root,listb,text,server
def main():
	server=sc.sichuanmj_server()
	server.restart(1)
	server.players[0].me.my_hand[0,:]=[0,0,0,1,2,2,2,3,3,3,4,4,4,5]
	server.common_info.peng[0,0,3]=1
	server.common_info.que[0]=2
	print(server.players[0].isGang(6,0))
	for i in range(4):
		server.dingque(i)
	server.input(0)

if __name__ == "__main__":
	main()