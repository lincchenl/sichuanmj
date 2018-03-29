import numpy as np
import split
import time
import gui
import mnn
import copy

#   私有元素
class sichuanmj_private:
	#自己的手牌状态
	#其中每张手牌除了包含手牌类型外，还附加有额外的信息，就是该手牌是牌库里的第几张摸到的
	#第一维第一个元素是手牌，第二维是第几张牌摸到的
	my_hand=None
	my_position=0
	def __init__(self):
		self.my_hand = np.zeros([2, 14], dtype=np.int)

	def __del__(self):
		self.my_hand=None
		self.my_position=None

#   公共元素
class sichuanmj_public:
	#牌库里面还有几张牌
	pool_cnt=0
	drop,status,hand_cnt,que,peng,gang=None,None,None,None,None,None
	#堂子信息
	#第三维是出牌玩家号
	def __init__(self):
		#第一维 0代表打的牌 1代表打牌的人
		self.drop=np.zeros([2,56],dtype=np.int)
		#四位玩家的公共视图
		self.status=np.zeros(4,dtype=np.int) #0 进行中，1 胡牌
		self.hand_cnt=np.zeros(4,dtype=np.int)
		self.que=np.zeros(4,dtype=np.int)
		self.peng=np.zeros([4,2,4],dtype=np.int)
		# 第二维0代表杠的牌，1代表第几手杠的，2代表杠的类型，3数代表放杠的人
		self.gang=np.zeros([4,4,4],dtype=np.int)#杠牌需要区分点杠0，暗杠1，补杠2，后补杠3（没有钱）

	def __del__(self):
		self.drop, self.status, self.hand_cnt, self.que, self.peng, self.gang = None, None, None, None, None, None

#   个人视图
class sichuanmj_client:
	me=None
	common_info=None
	# 当自己出牌时，记录可以杠而没杠的牌，免得每次摸牌都要重新遍历一遍
	gang_able=None
	gang_init=False
	ai=None
	actlist=None
	actlist_e=None
	decision_stack=None
	result_stack=None
	queue=None
	level=0
	def __init__(self):
		self.gang_able=[]
		self.actlist=np.zeros(6,dtype=np.int)
		self.actlist_e=np.zeros(14+14+18+1+3,dtype=np.int)
		self.decision_stack=[]
		self.result_stack=[]
		self.queue=[]

	def __del__(self):
		self.me = None
		self.common_info = None
		# 当自己出牌时，记录可以杠而没杠的牌，免得每次摸牌都要重新遍历一遍
		self.gang_able = None
		self.gang_init = False
		self.ai = None
		self.actlist = None
		self.actlist_e = None
		self.decision_stack = None
		self.result_stack = None
		self.queue=None

	# 把当前状态转化为神经网络的输入来喂ai
	def flatten_to_train(self):
		cnt=[self.me.my_hand.size,1,1,self.common_info.drop.size,self.common_info.status.size,
		     self.common_info.hand_cnt.size,self.common_info.que.size,self.common_info.peng.size,
		     self.common_info.gang.size,self.actlist_e.size]
		flatten=[self.me.my_hand.ravel(),self.me.my_position,self.common_info.pool_cnt,
		         self.common_info.drop.ravel(),self.common_info.status.ravel(),
		         self.common_info.hand_cnt.ravel(),self.common_info.que.ravel(),self.common_info.peng.ravel(),
		         self.common_info.gang.ravel(),self.actlist_e.ravel()*100]
		input=np.empty(sum(cnt),dtype=np.int)
		start,end=0,0
		for i in range(len(cnt)):
			start=end
			end+=cnt[i]
			input[start:end]=flatten[i]
		return input
	# 检查当前牌是否可碰，如果可碰返回该牌在手牌里的索引
	def isPeng(self,pai):
		pos=self.me.my_position
		if (pai-(self.common_info.que[pos]-1)*9-1)*(pai-self.common_info.que[pos]*9)<=0:
			return np.array([])
		index=np.argwhere(self.me.my_hand[0,:]==pai)
		if index.size>=2:
			return index
		else:
			return np.array([])
	# 检查是否可以杠牌，user是点杠人的编号（包括自己）
	# 返回所有可杠牌的list
	def isGang(self,pai,user):
		pos=self.me.my_position
		hand=self.me.my_hand[0,:]
		hand=hand[hand.nonzero()]
		peng=self.common_info.peng[pos,0,:]
		peng=peng[peng.nonzero()]
		if user==pos:
			#检查原生暗杠：
			if not self.gang_init:
				for i in 	hand:
					if np.argwhere(hand==i).size==4:
						if i not in self.gang_able: self.gang_able.append(i)
				self.gang_init=True
			if (pai - (self.common_info.que[pos] - 1) * 9 - 1) * (pai - self.common_info.que[pos] * 9) <= 0:
				return np.array(self.gang_able)
			# 检查手牌
			cnt = np.argwhere(hand == pai).size
			if cnt >= 3:
				if pai not in self.gang_able: self.gang_able.append(pai)
			#检查碰牌
			if pai in peng:
				if pai not in self.gang_able: self.gang_able.append(pai)
			#检查手上是否有未杠的牌
			for i in peng:
				if i in hand and (not i in self.gang_able): self.gang_able.append(i)
			return np.array(self.gang_able)
		else:
			if (pai - (self.common_info.que[pos] - 1) * 9 - 1) * (pai - self.common_info.que[pos] * 9) <= 0:
				return np.array([])
			# 检查手牌
			cnt = np.argwhere(self.me.my_hand[0, :] == pai).size
			if cnt >= 3:
				return np.array([pai])
			else:
				return np.array([])
	# 检查是否是花猪
	def isHuazhu(self):
		pos=self.me.my_position
		hand=self.me.my_hand[0,:]
		peng=self.common_info.peng[pos,0,:]
		gang=self.common_info.gang[pos,0,:]
		start=(self.common_info.que[pos]-1)*9+1
		end=start+8
		cnt=np.argwhere((hand-start)*(hand-end)<=0).size
		cnt+=np.argwhere((peng-start)*(peng-end)<=0).size
		cnt+=np.argwhere((gang-start)*(gang-end)<=0).size
		if cnt>0: return True
		return False
	# 检查是否胡牌或叫牌,user是放炮人的编号(包括自己）
	# 返回几番
	def isHu(self,pai,Gang=0):
		fan=0
		flag=False
		pos=self.me.my_position
		hand=self.me.my_hand[0,:]
		hand=hand[np.nonzero(hand)]
		peng=self.common_info.peng[pos,0,:]
		peng=peng[np.nonzero(peng)]
		gang=self.common_info.gang[pos,0,:]
		gang=gang[np.nonzero(gang)]
		# 判断是否花猪
		if self.isHuazhu(): return 0
		# 把牌的序列转化格式
		to_hu=[0 for i in range(34)]
		for i in hand:
			if i>0: to_hu[i-1]+=1
		to_hu[pai-1]+=1
		# 判断是否七对/龙七对
		if np.argwhere(peng>0).size+np.argwhere(gang>0).size==0:
			# 门清
			for i in range(len(to_hu[:27])):
				if to_hu[i]%2==1:break
			else:
				fan+=3
				flag=True
		# 除开七对只剩下普通胡牌
		if not flag:
			flag=split.get_hu_info(to_hu,34,34)
			if flag: fan=1
		# 计算额外的加番
		if not flag: return 0
		# 清一色
		wan=np.argwhere((hand-1)*(hand-9)<=0).size+np.argwhere((peng-1)*(peng-9)<=0).size\
		    +np.argwhere((gang-1)*(gang-9)<=0).size
		tiao=np.argwhere((hand-10)*(hand-18)<=0).size+np.argwhere((peng-10)*(peng-18)<=0).size\
		    +np.argwhere((gang-10)*(gang-18)<=0).size
		tong=np.argwhere((hand-19)*(hand-27)<=0).size+np.argwhere((peng-19)*(peng-27)<=0).size\
		    +np.argwhere((gang-19)*(gang-27)<=0).size
		tot=wan**2+tiao**2+tong**2
		if tot==wan**2 or tot==tiao**2 or tot==tong**2: fan+=2
		# 对对胡
		if to_hu.count(2)==1 and to_hu.count(1)==0:
			if to_hu.count(3)==0:
				#金钩吊
				fan+=2
			else:
				#对对胡
				fan+=1
		# 带根
		fan+=to_hu.count(4)
		fan+=np.argwhere(gang>0).size
		tmp=np.array(to_hu)
		tmp=tmp[peng[np.where(peng>0)]-1]
		fan+=np.argwhere(tmp==1).size
		# 杠上炮/杠上花
		if Gang==1: fan+=1
		# 海底捞月
		if self.common_info.pool_cnt==0: fan+=1
		return fan
	#接受并执行一个动作
	def act(self,actlist,server,Gang,no_from,pai):
		g=server.g
		pos=self.me.my_position
		valid=self.valid_oper(actlist)
		hand=self.me.my_hand[0,:]
		hand=hand[hand.nonzero()]
		peng=self.common_info.peng[pos,0,:]
		peng=peng[peng.nonzero()]
		gang=self.common_info.gang[pos,0,:]
		gang=gang[gang.nonzero()]
		if False:
			hint = self.ai_cal(valid)
			actno = g.update(server, actlist, self.me.my_position, valid,hint)
		else:
			if self.level<=0:
				actno=self.machine_choose(server,valid,Gang,no_from,pai,10)
				print(self.common_info.pool_cnt, self.common_info.que[pos], "(", pos, "<-", no_from, pai, ")",
						hand, peng,gang,valid[actno])
			else:
				actno=self.machine_choose_fast(valid,0.9)
		act_result=valid[actno]

		return act_result

	#列举出所有合法的操作
	def valid_oper(self,actlist):
		valid=[]
		pos=self.me.my_position
		hand=self.me.my_hand[0,:]
		hand=hand[hand.nonzero()]
		tmp=np.zeros(6,dtype=np.int)
		#最后四张必胡
		if self.common_info.pool_cnt<=4 and actlist[4]>0:
			tmp=np.array([0,0,0,0,actlist[4],0])
			valid.append((tmp.copy()))
			return valid
		if actlist[5]>0: #定缺
			tmp[5]=1
			valid.append(tmp.copy())
			tmp[5]=2
			valid.append(tmp.copy())
			tmp[5]=3
			valid.append(tmp.copy())
		if actlist[4]>0: #胡牌
			tmp.fill(0)
			tmp[4]=actlist[4]
			valid.append(tmp.copy())
		if actlist[3]>0: #杠牌
			tmp.fill(0)
			#判断是不是点杠
			if actlist[0]>0:
				#暗杠或者补杠
				for i in self.gang_able:
					tmp[3]=i
					valid.append(tmp.copy())
			else:
				#点杠
				tmp[3]=actlist[3]
				valid.append(tmp.copy())
		if actlist[2]>0: #碰牌
			tmp.fill(0)
			tmp[2]=actlist[2]
			valid.append(tmp.copy())
		if actlist[0]>0: #打牌
			tmp.fill(0)
			pai=actlist[0]
			start=(self.common_info.que[pos]-1)*9+1
			end=start+8
			index=np.argwhere((hand-start)*(hand-end)<=0)
			if index.size!=0 or (pai-start)*(pai-end)<=0: #花猪状态
				for i in hand[index]:
					tmp[0]=i
					valid.append(tmp.copy())
				if (pai-start)*(pai-end)<=0:
					tmp[0]=pai
					valid.append(tmp.copy())
			else:
				for i in hand:
					tmp[0]=i
					valid.append(tmp.copy())
				if pai<=27 and pai>=1:
					tmp[0]=pai
					valid.append(tmp.copy())
		tmp.fill(0)
		#只有不存在出牌和定缺的情况，才可以不采取任何操作
		if actlist[0]==0 and actlist[5]==0:
			tmp.fill(0)
			valid.append(tmp.copy())
		return valid

	def actlist_to_extended(self,actlist):
		pos=self.me.my_position
		hand=self.me.my_hand[0,:]
		peng=self.common_info.peng[pos,0,:]
		result=np.zeros(14+14+18+1+3,dtype=np.int)
		if actlist[0]>0:
			if not actlist[0] in hand:
				result[0]=1
			else:
				no=np.argwhere(hand==actlist[0])[0]
				result[no+1]=1
		elif actlist[2]>0:
			no=np.argwhere(hand==actlist[2])[0]
			result[14+no]=1
		elif actlist[3]>0:
			if actlist[3] in hand:
				no=np.argwhere(hand==actlist[3])[0]
				result[14 + 14 + no] = 1
			else:
				no=np.where(peng==actlist[3])[0]
				result[14 + 14 +14+ no] = 1
		elif actlist[4]>0:
			result[14+14+18]=1
		elif actlist[5]>0:
			result[14+14+18+1+actlist[5]-1]=1
		return result

	def machine_choose(self,server,valid,Gang,no_from,pai,cnt):
		valid_cnt=len(valid)
		expect=np.empty(valid_cnt,dtype=np.float)
		for j in range(valid_cnt):
			act=valid[j]
			bonus=0
			for i in range(cnt):
				bonus+=self.fork_server(server,act,Gang,no_from,pai)
			self.actlist[:]=act[:]
			self.actlist_e[:] = self.actlist_to_extended(self.actlist)
			self.decision_stack.append(self.flatten_to_train())
			expect[j]=bonus/cnt
			self.result_stack.append(bonus/cnt)
		#选择期望最高的
		index=np.argmax(expect)
		return index

	def machine_choose_fast(self,valid,ratio):
		cnt=len(valid)
		output=self.ai_cal(valid)
		if np.random.rand()>=ratio:
			#用softmax把赢钱的期望转化为选择的概率
			e=np.exp(output)
			e_sum=np.sum(e)
			index=np.random.choice(cnt,1,p=(e/e_sum))[0]
		else:
			index=np.argmax(output)
		return index

	def ai_cal(self,valid):
		cnt=len(valid)
		input=np.empty([cnt,300],dtype=np.float)
		for i in range(cnt):
			self.actlist_e[:]=self.actlist_to_extended(valid[i])
			input[i,:]=self.flatten_to_train()
		self.ai.input=input
		self.ai.forward(train=False)
		return self.ai.output[:,0]

	def fork_server(self,server,actlist,Gang,no_from,pai):
		sv=server.fork()
		pos = self.me.my_position
		next = pos
		next = sv.next_valid_player(sv.execute(actlist, next, Gang,no_from,pai))
		while next>=0:
			next=sv.next_valid_player(sv.input(next))
			if sv.common_info.pool_cnt==0: break
			if sv.common_info.status[pos]==0: break
			if np.argwhere(sv.common_info.status==1).size==3: break
		sv.endset()
		return sv.bonus[pos]

#   服务器视图
class sichuanmj_server:
	players=None
	common_info=None
	pool=None
	g=None
	checkpoint=0
	#杠牌收钱的记录，用来最后查叫的时候赔钱退钱用
	gang_stack=None
	master_cnt=0 #庄家
	step=-1
	bonus=None
	def __init__(self):
		#初始化时间
		self.checkpoint=time.time()
		#初始化ai
		ratio=0.001
		ai=mnn.mnn()
		fc1 = mnn.full_connect(300, 200)
		fc2 = mnn.full_connect(200, 200)
		fc3 = mnn.full_connect(200, 1)
		fc1.opti = mnn.adam(ratio)
		fc2.opti = mnn.adam(ratio)
		fc3.opti = mnn.adam(ratio)
		ai.addlayer(fc1)
		ai.addlayer(mnn.active_function(1)) #selu
		ai.addlayer(fc2)
		ai.addlayer(mnn.active_function(1)) #selu
		ai.addlayer(fc3)
		ai.addlayer(mnn.active_function(1))
		#读取已经训练的数据
		#ai=ai.load("d:\\paras\\mj.pkl")
		#初始化公共视图
		self.common_info=sichuanmj_public()
		self.players=np.zeros(4,dtype=sichuanmj_client)
		self.pool=None
		self.bonus=None
		self.g=gui.simple_gui()
		self.gang_stack=[]

		#初始化四个玩家
		for i in range(4):
			self.players[i]=sichuanmj_client()
			self.players[i].me=sichuanmj_private()
			self.players[i].common_info=self.common_info
			self.players[i].me.my_position=i
			self.players[i].level=0
			self.players[i].ai=ai #四个小碧池共享一个ai

	def __del__(self):
		self.players[0] = None
		self.players[1] = None
		self.players[2] = None
		self.players[3] = None
		self.common_info = None
		self.pool = None
		self.g = None
		self.gang_stack = None
		self.bonus = None

	#重新开始一局牌
	def restart(self,master):
		self.master_cnt=master
		#初始化牌库
		self.bonus=np.zeros(4,dtype=np.int)
		self.pool=np.zeros(108, dtype=np.int)
		for i in range(1,28): #一共有27种牌，每种牌4张，分别是1-9万，1-9条，1-9筒
			for j in range(4):
				self.pool[4*(i-1)+j]=i
		self.common_info.pool_cnt=self.pool.size

		#初始化公共视图
		self.common_info.peng.fill(0)
		self.common_info.gang.fill(0)
		self.common_info.drop.fill(0)
		self.common_info.hand_cnt.fill(0)
		self.common_info.que.fill(0)
		self.common_info.status.fill(0)
		self.bonus.fill(0)
		self.gang_stack=[]
		self.step=-1

		#初始化私人视图
		for i in range(4):
			self.players[i].me.my_hand.fill(0)
			self.players[i].gang_able=[]
			self.players[i].gang_init=False
			self.players[i].decision_stack=[]
			self.players[i].result_stack = []
		self.mopai()

	#摸n张牌
	def fetch(self,count):
		seed=int(time.time())
		seed+=self.pool.size
		np.random.seed(seed)
		label=np.random.choice(self.pool.size,size=count,replace=False)
		result=self.pool[label]
		self.pool=np.delete(self.pool,label)
		self.common_info.pool_cnt=self.pool.size
		return result

	#摸牌阶段
	def mopai(self):
		for i in range(4):
			hand=self.players[i].me.my_hand
			hand.fill(0)
			hand[0,1:14]=self.fetch(13)
			self.common_info.hand_cnt[i]=13
			index=hand[0,:].argsort()
			self.players[i].me.my_hand=hand[:,index]

	#定缺
	def dingque(self,player_cnt):
		#actlist = np.zeros(6, dtype=np.int)
		#actlist[5]=1
		#actlist=self.players[player_cnt].act(actlist,self)
		#index=np.argwhere(actlist>0)
		#if index.size==1 :
		#	if index[0]==5 and (actlist[5]-4)*(actlist[5]-1)<=0:
		#		self.common_info.que[player_cnt]=actlist[5]
		#	else:
		#		print("命令错误！！！")
		#		exit(-1)
		hand=self.players[player_cnt].me.my_hand[0,:]
		hand=hand[hand.nonzero()]
		flag=False
		#手牌是有序的
		for i in range(hand.size):
			if hand[i]>=10:
				flag=True
				break
		if flag:
			cnt_wan=i
		else:
			cnt_wan=hand.size

		flag=False
		for i in range(cnt_wan,hand.size):
			if hand[i]>=19:
				flag=True
				break
		if flag:
			cnt_tiao=i-cnt_wan
			cnt_tong=hand.size-i
		else:
			if cnt_wan==hand.size:
				cnt_tiao=0
				cnt_tong=0
			else:
				cnt_tiao=i-cnt_wan
				cnt_tong=0
		wan=hand[:cnt_wan]
		tiao=hand[cnt_wan:cnt_wan+cnt_tong]
		tong=hand[cnt_wan+cnt_tong:]
		for i in wan:
			size=np.argwhere(wan==i).size
			if size>2:	cnt_wan+=size-1
		for i in tiao:
			size=np.argwhere(tiao==i).size
			if size>2:	cnt_tiao+=size-1
		for i in tong:
			size=np.argwhere(tong==i).size
			if size>2:	cnt_tong+=size-1

		cnt_min=min(cnt_wan,cnt_tiao,cnt_tong)
		if (cnt_min==cnt_wan):
			self.common_info.que[player_cnt]=1
		elif (cnt_min==cnt_tiao):
			self.common_info.que[player_cnt]=2
		else:
			self.common_info.que[player_cnt]=3

	def error(self,p_no,err_no):
		print("命令错误！！！,错误号：",err_no)
		print("造成错误的用户：",p_no)
		exit(-1)

	# 摸牌-》自己能否杠或者胡，
	def input(self,player_cnt,Gang=0):
		# actlist定义：允许执行的动作列表，包含6个元素，
		# 分别代表：打牌，吃牌，碰牌，杠牌，胡牌，定缺
		# 服务器端给客户端的actlist哪一项为1代表哪一项允许执行，为0代表不允许
		if self.common_info.pool_cnt==0: return -1
		flag_gang=0
		self.step+=1
		actlist=np.zeros(6,dtype=np.int)
		pai=self.fetch(1)[0]
		me=self.players[player_cnt]
		me.queue=[]
		check_list=[]
		#判断是否可以杠或者胡牌
		fan=me.isHu(pai,Gang)
		if me.isGang(pai,player_cnt).size >0 and self.common_info.pool_cnt>0: flag_gang=1
		allow=[0]
		if fan>0: allow.append(4)
		if flag_gang>0 : allow.append(3)
		actlist[0],actlist[3],actlist[4]=pai,flag_gang,fan
		actlist[:]=self.players[player_cnt].act(actlist,self,Gang,player_cnt,pai)
		self.validate_actlist(player_cnt,actlist,allow)
		index=np.argwhere(actlist>0)
		if index.size==0: self.error(player_cnt,92)
		if index[0]==3:
			a_back=actlist.copy()
			if not actlist[3] in self.common_info.peng[player_cnt,0,:]:   #暗杠
				if pai==actlist[3]:
					self.del_hand(player_cnt,pai,3)
					self.add_gang(player_cnt,player_cnt,1,pai)
				else:
					self.del_hand(player_cnt, actlist[3], 4)
					self.add_gang(player_cnt, player_cnt, 1, actlist[3])
					self.add_hand(player_cnt,pai)
				self.gangpai(player_cnt,player_cnt,1)
			else :  #补杠
				#抢杠
				# 剔除自己和已经胡牌的人，参照自己的顺序建立一个名单
				for i in range(player_cnt + 1, player_cnt + 4):
					real_i = i % 4
					if self.common_info.status[real_i] == 0: check_list.append(real_i)
				#逐个检查是否胡牌
				for i in check_list:
					if i==player_cnt or self.common_info.status[i]==1: continue
					fan = self.players[i].isHu(actlist[3],1)
					if fan>0:
						me.queue.append(np.array([0,0,0,0,fan,0,player_cnt,pai],dtype=np.int))
				#把杠牌放入到队列的最后一个任务
				me.queue.append(np.array([0,0,0,actlist[3],0,0,player_cnt,pai]))
				#处理队列中的胡牌请求
				next=-1
				while me.queue:
					act=me.queue.pop(0)
					my_cnt=act[6]
					pai=act[7]
					if act[4]<=0: break
					a_back[:]=self.players[my_cnt].act(act[:6],self,1,player_cnt,pai)
					self.validate_actlist(my_cnt,a_back,[4])
					if a_back[4]>0:    #胡牌
						self.hupai(my_cnt,player_cnt,fan,actlist[3])
						#只要有一个人胡牌，就不再允许补杠
						if me.queue[-1][4]<=0: me.queue.pop(-1)
						next=my_cnt
				if next>=0: return next
				if actlist[3] == pai:   #有钱的补杠
					self.add_gang(player_cnt,player_cnt,2,pai)
					self.del_peng(player_cnt,pai)
					self.gangpai(player_cnt,player_cnt,2)
				else:
					self.add_gang(player_cnt,player_cnt,3,actlist[3])
					self.del_peng(player_cnt,actlist[3])
					self.del_hand(player_cnt,actlist[3],1)
					self.add_hand(player_cnt,pai)
					self.gangpai(player_cnt,player_cnt,3)
			me.gang_able.remove(actlist[3])
			return self.input(player_cnt,1)
		elif index[0]==4:
				self.hupai(player_cnt,player_cnt,fan,pai)
				return player_cnt
		elif index[0]==0:
				if pai!=actlist[0]:
					self.del_hand(player_cnt,actlist[0], 1)
					self.add_hand(player_cnt,pai)
				#出牌
				return self.output(actlist[0],player_cnt, Gang)


	# 打牌-》判断别人是否碰，杠或者胡
	def output(self,pai,player_cnt,Gang=0):
		me=self.players[player_cnt]
		actlist=np.zeros(8,dtype=np.int)
		next=-1
		check_list=[]
		self.players[player_cnt].queue=[]
		queue=self.players[player_cnt].queue
		# 把能杠的牌打出
		if pai in me.gang_able: me.gang_able.remove(pai)
		# 剔除自己和已经胡牌的人，参照自己的顺序建立一个名单
		for i in range(player_cnt+1,player_cnt+4):
			real_i=i%4
			if self.common_info.status[real_i] == 0: check_list.append(real_i)
		#建立动作队列
		# 放炮
		for i in check_list:
			fan=self.players[i].isHu(pai,Gang)
			if fan>0:
				actlist.fill(0)
				actlist[4]=fan
				actlist[6]=i
				actlist[7]=pai
				queue.append(actlist.copy())
		# 碰
		for i in check_list:
			index=self.players[i].isPeng(pai)
			if index.size>0:
				actlist.fill(0)
				actlist[2]=pai
				actlist[6]=i
				actlist[7]=pai
				queue.append(actlist.copy())
		# 杠
		for i in check_list:
			index=self.players[i].isGang(pai,player_cnt)
			if index.size>0 and self.common_info.pool_cnt>0:
				actlist.fill(0)
				actlist[3]=pai
				actlist[6]=i
				actlist[7]=pai
				queue.append(actlist.copy())
		#执行动作队列
		while queue:
			act=queue.pop(0)
			#先看是否有人胡牌
			if act[4]>0:
				actlist=self.players[act[6]].act(act[0:6],self,Gang,player_cnt,pai)
				self.validate_actlist(act[6],actlist,[4])
				if actlist[4]>0:
					self.hupai(act[6], player_cnt, actlist[4], pai)
					next= act[6]
					#只要有一个人胡了就将队列里的所有碰和杠的队员删除
					self.players[player_cnt].queue = [ele for ele in queue if ele[4]>0]
					queue=self.players[player_cnt].queue
			#碰牌
			if act[2]>0:
				actlist=self.players[act[6]].act(act[0:6],self,Gang,player_cnt,pai)
				self.validate_actlist(player_cnt,actlist,[2])
				if actlist[2]>0:
					self.add_peng(act[6], pai)
					self.del_hand(act[6], pai, 2)
					self.players[player_cnt].queue=[]
					queue=[]
					actlist = self.players[act[6]].act(np.array([99,0,0,0,0,0],dtype=np.int), self, Gang, player_cnt, pai)
					self.validate_actlist(act[6],actlist, [0])
					if actlist[0]<=0: self.error(act[6],91)
					self.del_hand(act[6],actlist[0],1)
					return self.output(actlist[0],act[6])
			#杠牌
			if act[3]>0:
				actlist = self.players[act[6]].act(act[0:6], self, Gang, player_cnt, pai)
				self.validate_actlist(player_cnt,actlist, [3])
				if actlist[3] > 0:
					self.add_gang(act[6], player_cnt, 0, pai)
					self.del_hand(act[6], pai, 3)
					# 奖金结算
					self.gangpai(act[6], player_cnt, 0)
					self.players[player_cnt].queue=[]
					queue=[]
					return self.input(act[6],1)
		#有人胡了牌就直接返回
		if next>=0: return next
		#出牌
		#放入堂子中
		self.common_info.drop[0,self.step]=pai
		self.common_info.drop[1, self.step] = player_cnt
		return player_cnt

	def del_hand(self,player_cnt,pai,cnt):
		hand=self.players[player_cnt].me.my_hand
		index=np.argwhere(hand[0,:]==pai)
		if index.size<cnt: self.error(player_cnt,101)
		index=index[0][0]
		hand[:,index:index+cnt].fill(0)
		index=hand[0,:].argsort()
		hand[:,:]=hand[:,index]

	def add_hand(self,player_cnt,pai):
		hand=self.players[player_cnt].me.my_hand
		hand[:,0]=[pai,self.step]
		index=hand[0,:].argsort()
		hand[:,:]=hand[:,index]

	def del_peng(self,player_cnt,pai):
		peng=self.common_info.peng[player_cnt,:,:]
		index=np.argwhere(peng[0,:]==pai)
		if index.size==0: self.error(player_cnt,102)
		index=index[0][0]
		peng[:,index].fill(0)
		index=peng[0,:].argsort()
		peng[:,:]=peng[:,index]

	def add_peng(self,player_cnt,pai):
		peng=self.common_info.peng[player_cnt,:,:]
		peng[:,0]=[pai,self.step]
		index=peng[0,:].argsort()
		peng[:,:]=peng[:,index]

	def add_gang(self,player_cnt,player_from,kind,pai):
		gang=self.common_info.gang[player_cnt,:,:]
		gang[:,0]=[pai,self.step,kind,player_from]
		index=gang[0,:].argsort()
		gang[:,:]=gang[:,index]

	def next_valid_player(self,player_cnt):
		next = (player_cnt + 1) % 4
		while self.common_info.status[next] == 1:
			next = (next + 1) % 4
			if next==player_cnt: return next
		return next

	def validate_actlist(self,player_cnt,actlist,allow):
		index=np.argwhere(actlist>0)
		if index.size>1: self.error(player_cnt,90) #命令多于一个
		if index.size==0: return
		if index.size==1 and (not index[0] in allow):
			self.error(player_cnt,91)

	def jiesuan(self,player,money):
		self.bonus[player]+=money
	def hupai(self,p_hu,p_from,fan,pai):
		self.add_hand(p_hu,pai)
		self.common_info.status[p_hu]=1
		base=2**min(3,fan-1)  #3番封顶
		if p_hu!=p_from: #放炮
			self.jiesuan(p_hu,base)
			self.jiesuan(p_from, -base)
		else:
			base+=1 #自摸加底
			cnt=0
			for i in range(4):
				if i==p_hu or self.common_info.status[i]==1: continue
				cnt+=1
				self.jiesuan(i,-base)
			self.jiesuan(p_hu, cnt*base)
	def gangpai(self,p_gang,p_from,kind):
		if kind==0: #点杠
			self.gang_stack.append([p_gang,p_from,2])
			self.jiesuan(p_gang,2)
			self.jiesuan(p_from,-2)
		elif kind==1:   #暗杠
			cnt=0
			for i in range(4):
				if i!=p_gang or self.common_info.status[i]==1: continue
				cnt+=1
				self.gang_stack.append([p_gang,i,2])
				self.jiesuan(i,-2)
			self.jiesuan(p_gang,cnt*2)
		elif kind==2:
			cnt=0
			for i in range(4):
				if i!=p_gang or self.common_info.status[i]==1: continue
				cnt+=1
				self.gang_stack.append([p_gang,i,1])
				self.jiesuan(i,-1)
			self.jiesuan(p_gang,cnt*1)

	def execute(self,act,player_cnt,Gang,no_from,pai):
		actlist=copy.deepcopy(act)
		me = self.players[player_cnt]
		if no_from==player_cnt:
			if actlist[3]>0:
				a_back=actlist.copy()
				if not actlist[3] in self.common_info.peng[player_cnt,0,:]:   #暗杠
					if pai==actlist[3]:
						self.del_hand(player_cnt,pai,3)
						self.add_gang(player_cnt,player_cnt,1,pai)
					else:
						self.del_hand(player_cnt, actlist[3], 4)
						self.add_gang(player_cnt, player_cnt, 1, actlist[3])
						self.add_hand(player_cnt,pai)
					self.gangpai(player_cnt,player_cnt,1)
				else :  #补杠
					#抢杠
					rel_max=0
					for i in range(4):
						if i==player_cnt or self.common_info.status[i]==1: continue
						fan = self.players[i].isHu(actlist[3],1)
						if fan>0:
							a_back.fill(0)
							a_back[4]=fan
							a_back[:]=self.players[i].act(a_back,self,Gang,no_from,pai)
							self.validate_actlist(i,a_back,[4])
							if a_back[4]>0:    #胡牌
								self.hupai(i,player_cnt,fan,actlist[3])
								rel_pos=(i-player_cnt)%4
								if rel_pos>rel_max: rel_max=rel_pos
					if rel_max>0:
						me.gang_able.remove(actlist[3])
						rel_max=(rel_max+player_cnt)%4
						return rel_max
					if actlist[3] == pai:   #有钱的补杠
						self.add_gang(player_cnt,player_cnt,2,pai)
						self.del_peng(player_cnt,pai)
						self.gangpai(player_cnt,player_cnt,2)
					else:
						self.add_gang(player_cnt,player_cnt,3,actlist[3])
						self.del_peng(player_cnt,actlist[3])
						self.del_hand(player_cnt,actlist[3],1)
						self.add_hand(player_cnt,pai)
						self.gangpai(player_cnt,player_cnt,3)
				me.gang_able.remove(actlist[3])
				return self.input(player_cnt,1)
			elif actlist[4]>0:
				self.hupai(player_cnt,player_cnt,actlist[4],pai)
				return player_cnt
			elif actlist[0]>0:
				if pai!=actlist[0]:
					self.del_hand(player_cnt,actlist[0], 1)
					self.add_hand(player_cnt,pai)
				#出牌
				return self.output(actlist[0],player_cnt, Gang)
		else:
			queue=self.players[no_from].queue
			next=-1
			if actlist[4] > 0:
				self.hupai(player_cnt, no_from, actlist[4], pai)
				next=player_cnt
				self.players[no_from].queue = [ele for ele in queue if ele[4] > 0]
				queue = self.players[no_from].queue
			if actlist[2] > 0:
				self.add_peng(player_cnt, pai)
				self.del_hand(player_cnt, pai, 2)
				self.players[no_from].queue=[]
				queue=[]
				actlist = self.players[player_cnt].act(np.array([99, 0, 0, 0, 0, 0], dtype=np.int), self, Gang,
				                                   no_from, 99)
				self.validate_actlist(player_cnt,actlist, [0])
				if actlist[0] <= 0: self.error(act[6], 91)
				self.del_hand(player_cnt,actlist[0],1)
				return self.output(actlist[0],player_cnt)
			if actlist[3] > 0:
				self.add_gang(player_cnt, no_from, 0, pai)
				self.del_hand(player_cnt, pai, 3)
				# 奖金结算
				self.gangpai(player_cnt, no_from, 0)
				self.players[no_from].queue=[]
				queue=[]
				return self.input(player_cnt, 1)
			if actlist[0] > 0:
				self.del_hand(player_cnt,actlist[0],1)
				return self.output(pai,player_cnt)
			#只要有一个人进行了碰，杠，胡之中的一个动作，牌就不会被放到牌池中

			# 执行动作队列
			while queue:
				act = queue.pop(0)
				pai = act[7]
				me = act[6]
				# 先看是否有人胡牌
				if act[4] > 0:
					actlist = self.players[me].act(act[0:6], self, Gang, no_from, pai)
					self.validate_actlist(me,actlist, [4])
					if actlist[4] > 0:
						self.hupai(me, no_from, actlist[4], pai)
						next = me
						# 只要有一个人胡了就将队列里的所有碰和杠的队员删除
						self.players[player_cnt].queue = [ele for ele in queue if ele[4] > 0]
						queue = self.players[player_cnt].queue
				# 碰牌
				if act[2] > 0:
					actlist = self.players[me].act(act[0:6], self, Gang, no_from, pai)
					self.validate_actlist(me,actlist, [2])
					if actlist[2] > 0:
						self.add_peng(me, pai)
						self.del_hand(me, pai, 2)
						self.players[no_from].queue=[]
						queue = []
						actlist = self.players[me].act(np.array([99, 0, 0, 0, 0, 0], dtype=np.int), self, Gang,
						                                   no_from, 99)
						self.validate_actlist(me,actlist, [0])
						if actlist[0] <= 0: self.error(me, 91)
						self.del_hand(me,actlist[0],1)
						return self.output(actlist[0],me)
				#杠牌
				if act[3] > 0:
					actlist = self.players[me].act(act[0:6], self, Gang, no_from, pai)
					self.validate_actlist(me,actlist, [3])
					if actlist[3] > 0:
						if actlist[3] not in self.common_info.peng[me,0,:]: #点杠
							self.add_gang(me, no_from, 0, pai)
							self.del_hand(me, pai, 3)
							# 奖金结算
							self.gangpai(me,no_from, 0)
						elif pai==actlist[3]:       #有钱的补杠
							self.add_gang(me, no_from, 2, pai)
							self.del_peng(me, pai)
							self.gangpai(me, no_from, 2)
							self.players[me].gang_able.remove(actlist[3])
						else:
							self.add_gang(me, no_from, 3, actlist[3])
							self.del_peng(me, actlist[3])
							self.del_hand(me, actlist[3], 1)
							self.add_hand(me, pai)
							self.gangpai(me, no_from, 3)
							self.players[me].gang_able.remove(actlist[3])
						return self.input(me, 1)
			if next>=0: return next
			# 出牌
			# 放入堂子中
			self.common_info.drop[0, self.step] = pai
			self.common_info.drop[1, self.step] = player_cnt
			return player_cnt

	def endset(self):

		flag=False
		#三个人都胡牌了，不用查叫查花猪
		if np.argwhere(self.common_info.status==1).size>=3: flag=True
		# 检查谁有叫
		if not flag:
			change=False
			good=[]
			bad=[]
			hua=[]
			mfan=[0,0,0,0]
			for i in range(4):
				if self.common_info.status[i]==1: continue
				if self.players[i].isHuazhu():
					hua.append(i)
					bad.append(i)
					return
				maxfan=0
				for j in range(1,28):
					que=self.common_info.que[i]
					if j>=9*(que-1)+1 and j<=9*(que-1)+9: continue
					fan=self.players[i].isHu(j)
					if fan>maxfan: maxfan=fan
				if maxfan>0:
					good.append(i)
					mfan[i]=maxfan-1-1  #计算番数的时候从1番开始，且当牌池为0时处理成海底捞月加番，所以还要再减1
				else:
					bad.append(i)
			#花猪赔所有
			for i in hua:
				change=True
				self.bonus[i]-=2**3
				for j in range(4):
					if j!=i: self.bonus[j]+=2**3
			#没叫赔有叫
			for i in bad:
				for j in good:
					change = True
					self.bonus[i]-=2**min(3,mfan[j])
					self.bonus[j]+=2**min(3,mfan[j])
			#没叫返杠钱
			for i in bad:
				for record in self.gang_stack:
					if record[0]==i:
						change=True
						self.bonus[i]-=record[2]
						self.bonus[record[1]]+=record[2]
			if self.players[0].level==0:
				print("本局流局！！！,输赢结果是：",self.bonus)
				if hua: print("花猪有：",hua)
				if bad: print("没下叫的有：",bad)
		else:
			if self.players[0].level == 0:
				print("本局提前结束！最后输赢结果是：", self.bonus)

	def feed_ai(self,filename,iter,player):
		cnt=len(self.players[player].decision_stack)
		tot_input=np.empty([cnt,300],dtype=np.int)
		tot_output=np.empty([cnt,1],dtype=np.float)
		for i in range(cnt):
			p=self.players[player].decision_stack[i]
			q=self.players[player].result_stack[i]
			tot_input[i,:]=p[:]
			tot_output[i,0]=q
		self.players[0].ai.input=tot_input
		for i in range(iter):
			self.players[0].ai.forward(True,tot_output)
			self.players[0].ai.backward()
		if True:#time.time()>self.checkpoint+600 :    #10分钟保存一次
			self.players[0].ai.save(filename)
			self.checkpoint=time.time()

	#完全复制一个server对象，它们的ai，gui可以共享。但是必须完全复制它们的每一个元素
	def fork(self):
		gui=self.g
		ai=self.players[0].ai
		self.g=None
		for i in range(4):
			self.players[i].ai=None
		cp=copy.deepcopy(self)
		for i in range(4):
			cp.players[i].ai=ai
			self.players[i].ai=ai
			cp.players[i].level=self.players[i].level+1
		self.g=gui
		return cp
