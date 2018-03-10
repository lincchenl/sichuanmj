# -*-coding:utf-8-*-
import sichuanmj as sc
import numpy as np
global root,listb,text,server
def main():
	server=sc.sichuanmj_server()
	while True:
		server.restart(0)
		for i in range(4):
			server.dingque(i)
		next = server.master_cnt
		while next>=0:
			next=server.next_valid_player(server.input(next))
			if server.common_info.pool_cnt==0: break
			if np.argwhere(server.common_info.status==1).size==3: break
		server.endset()



if __name__ == "__main__":
	main()