# -*-coding:utf-8-*-
import sichuanmj as sc
import numpy as np
global root,listb,text,server
def main():
	server=sc.sichuanmj_server()
	tot_cnt=0
	valid_cnt=0
	while True:
		tot_cnt+=1
		print ("开始第",tot_cnt,"次训练","之前有用的训练有",valid_cnt,"次")
		server.restart(0)
		for i in range(4):
			server.dingque(i)
		next = server.master_cnt
		while next>=0:
			next=server.next_valid_player(server.input(next))
			if server.common_info.pool_cnt==0: break
			if np.argwhere(server.common_info.status==1).size==3: break
		server.endset()
		if np.sum(server.bonus**2)==0: continue
		valid_cnt+=1
		print("正在学习上一局牌谱。。。")
		for i in range(4):
			if server.players[i].decision_stack:
				server.feed_ai( "d:\\paras\\checkpoint.pkl", 5, i)



if __name__ == "__main__":
	main()