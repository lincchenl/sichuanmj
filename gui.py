# -*-coding:utf-8-*-
from tkinter import *
import sichuanmj as sc
class simple_gui():
	root=None
	list=None
	list2=None
	bt=None
	myno=0
	def __init__(self):
		# 初始化图形界面：
		self.root = Tk()
		self.list = Listbox(self.root, width=90, height=5)
		self.list2=Listbox(self.root, width=90, height=20)
		self.bt = Button(self.root,width=10, text="执行", state=DISABLED, command=self.getstr)
		self.list.grid(row=0, column=0, columnspan=35, sticky=W)
		self.bt.grid(row=4)
		self.list2.grid(row=5, column=0, columnspan=35, sticky=W)
	def update(self,server,actlist,user,valid):
		self.list.delete(0, END)
		self.list2.delete(0,END)
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
			if i==user and actlist[0]>=1 and actlist[0]<=27:
				str_mo='摸牌：'+self.arr_to_str([actlist[0]])
				string = str_hand+str_mo+str_peng+str_gang+str_que
			else:
				string = str_hand + str_peng + str_gang + str_que

			self.list.insert(i, string)
		for i in valid:
			string=self.translate(i)
			self.list2.insert(END,string)
		#显示命令
		self.list.select_set(user)
		self.bt.configure(state=ACTIVE)
		self.root.mainloop()
		return self.myno
	def getstr(self):
		sel=self.list2.curselection()
		if len(sel)==1:
			self.myno=sel[0]
			print (self.list2.get(self.myno))
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
		if actlist[0]>0:
			string="出牌："+dict[actlist[0]]
		if actlist[2]>0:
			string = "碰牌：" + dict[actlist[2]]
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





