# -*-coding:utf-8-*-
from tkinter import *
import sichuanmj as sc
class simple_gui():
	root=None
	list=None
	user=[None,None,None,None]
	drop=None
	bt=None
	myno=0
	def __init__(self):
		# 初始化图形界面：
		self.root = Tk()
		self.list = Listbox(self.root, width=90, height=20)
		for i in range(4):
			self.user[i]=Label(self.root,width=90,height=1,anchor=W,bg="#808000")
			self.user[i].grid(row=i,column=0,columnspan=90,sticky=W)
		self.drop = Label(self.root, width=90, height=4,anchor=NW,bg='#6495ED')
		self.bt = Button(self.root,width=10, text="执行", state=DISABLED, command=self.getstr)
		self.drop.grid(row=4,column=0,columnspan=90,sticky=W)
		self.list.grid(row=8, column=0, columnspan=35, sticky=W)
		self.bt.grid(row=9)

	def update(self,server,actlist,user,valid,hint):
		self.list.delete(0, END)
		# 显示手牌
		for i in range(4):
			me = server.players[i]
			hand = me.me.my_hand[0, :]
			hand = hand[hand.nonzero()]
			str_hand=' 手牌：'+self.arr_to_str(hand)
			peng = server.common_info.peng[i, 0, :]
			peng = peng[peng.nonzero()]
			str_peng=' 碰牌：'+self.arr_to_str(peng)
			gang = server.common_info.gang[i, 0, :]
			gang = gang[gang.nonzero()]
			str_gang=' 杠牌：'+self.arr_to_str(gang)
			que = server.common_info.que[i]
			str_que=' 缺门：'+str(que)
			if i!=0: str_hand=""
			if i==user and actlist[0]>=1 and actlist[0]<=27:
				str_mo='摸牌：'+self.arr_to_str([actlist[0]])
				string = str_hand+str_mo+str_peng+str_gang+str_que
			else:
				string = str_hand + str_peng + str_gang + str_que
			self.user[i].configure(text=string)
		drop_arr=server.common_info.drop[0,:]
		drop_arr=drop_arr[drop_arr.nonzero()]
		arr1=drop_arr[0:14]
		arr2=drop_arr[14:28]
		arr3=drop_arr[28:42]
		arr4=drop_arr[42:56]
		string=self.arr_to_str(arr1)+"\n"+ self.arr_to_str(arr2)+"\n"+ \
		       self.arr_to_str(arr3) + "\n" +self.arr_to_str(arr4)
		self.drop.configure(text=string)
		for i in range(4):
			if i==user:
				self.user[i].configure(fg='RED')
			else:
				if server.common_info.status[i]==1:
					self.user[i].configure(fg='GREEN')
				else:
					self.user[i].configure(fg='BLUE')

		for i in range(len(valid)):
			string=self.translate(valid[i])+"赢钱期望："+str(hint[i])
			self.list.insert(END,string)
		#显示命令
		self.bt.configure(state=ACTIVE)
		self.root.mainloop()
		return self.myno
	def getstr(self):
		sel=self.list.curselection()
		if len(sel)==1:
			self.myno=sel[0]
			print (self.list.get(self.myno))
			self.root.quit()

	def arr_to_str(self,arr):
		result=""
		dict={1:'一万',2:'二万',3:'三万',4:'四万',5:'五万',6:'六万',7:'七万',8:'八万',9:'九万',
		      10: '一条', 11: '二条', 12: '三条', 13: '四条', 14: '五条', 15: '六条', 16: '七条', 17: '八条', 18: '九条',
		      19: '一筒', 20: '二筒', 21: '三筒', 22: '四筒', 23: '五筒', 24: '六筒', 25: '七筒', 26: '八筒', 27: '九筒'}
		for i in arr:
			result=result+dict[i]+' '
		return result

	def translate(self,actlist):
		dict={1:'一万',2:'二万',3:'三万',4:'四万',5:'五万',6:'六万',7:'七万',8:'八万',9:'九万',
		      10: '一条', 11: '二条', 12: '三条', 13: '四条', 14: '五条', 15: '六条', 16: '七条', 17: '八条', 18: '九条',
		      19: '一筒', 20: '二筒', 21: '三筒', 22: '四筒', 23: '五筒', 24: '六筒', 25: '七筒', 26: '八筒', 27: '九筒'}
		string="放弃，什么也不做"
		if actlist[2]>0:
			string = "碰牌：" + dict[actlist[2]]+" 之后出牌："+dict[actlist[0]]
		if actlist[0]>0 and actlist[2]==0:
			string="出牌："+dict[actlist[0]]
		if actlist[3]>0:
			string = "杠牌：" + dict[actlist[3]]
		if actlist[4]>0:
			string = "胡牌：" + str(actlist[4]) + " 番"
		if actlist[5]>0:
			if actlist[5]==1:
				string = "定缺万子"
			if actlist[5]==2:
				string = "定缺条子"
			if actlist[5]==3:
				string = "定缺筒子"
		return string





