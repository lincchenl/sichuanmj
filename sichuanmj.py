import numpy as np
import split
import time
import gui

#   私有元素
class sichuanmj_private:
	#自己的手牌状态
	#其中每张手牌除了包含手牌类型外，还附加有额外的信息，就是该手牌是牌库里的第几张摸到的
	#第一维第一个元素是手牌，第二维是第几张牌摸到的
	my_hand=None
	my_position=0
	def __init__(self):
		self.my_hand = np.zeros([2, 14], dtype=np.int)

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

#   个人视图
class sichuanmj_client:
	me=None
	common_info=None
	# 当自己出牌时，记录可以杠而没杠的牌，免得每次摸牌都要重新遍历一遍
	gang_able=None
	gang_init=False
	def __init__(self):
		self.gang_able=[]
		self.actlist=np.zeros(6,dtype=np.int)

	# 把当前状态转化为神经网络的输入来喂ai
	def flatten_to_train(self):
		cnt=[self.me.my_hand.size,1,1,self.common_info.drop.size,self.common_info.status.size,
		     self.common_info.hand_cnt.size,self.common_info.que.size,self.common_info.peng.size,
		     self.common_info.gang.size,self.actlist.size]
		flatten=[self.me.my_hand.ravel(),self.me.my_position,self.common_info.pool_cnt,
		         self.common_info.drop.ravel(),self.common_info.status.ravel(),
		         self.common_info.hand_cnt.ravel(),self.common_info.que.ravel(),self.common_info.peng.ravel(),
		         self.common_info.gang.ravel(),self.actlist.ravel()]
		input=np.empty(sum(cnt),dtype=np.float)
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
			cnt = np.argwhere(self.me.my_hand[0, :] == pai).size
			if cnt >= 3:
				self.gang_able.append(pai)
			#检查碰牌
			cnt=np.argwhere(self.common_info.peng[pos,0,:]==pai).size
			if cnt>0:
				self.gang_able.append(pai)
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
		if self.isHuazhu(): return False
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
				fan+=2
				flag=True
		# 除开七对只剩下普通胡牌
		if not flag:
			flag=split.get_hu_info(to_hu,34,34)
			if flag: fan+=1
		# 计算额外的加番
		if not flag: return 0
		print (to_hu[0:27])
		# 清一色
		wan=np.argwhere((hand-1)*(hand-9)<=0).size+np.argwhere((peng-1)*(peng-9)<=0).size\
		    +np.argwhere((gang-1)*(gang-9)<=0).size
		tiao=np.argwhere((hand-10)*(hand-18)<=0).size+np.argwhere((peng-10)*(peng-18)<=0).size\
		    +np.argwhere((gang-10)*(gang-18)<=0).size
		tong=np.argwhere((hand-19)*(hand-27)<=0).size+np.argwhere((peng-19)*(peng-27)).size\
		    +np.argwhere((gang-19)*(gang-27)).size
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
	def act(self,actlist,server):
		g=server.g
		valid=self.valid_oper(actlist)
		actno=g.update(server,actlist,self.me.my_position,valid)
		return valid[actno]
	#列举出所有合法的操作
	def valid_oper(self,actlist):
		valid=[]
		pos=self.me.my_position
		hand=self.me.my_hand[0,:]
		tmp=np.zeros(6,dtype=np.int)
		if actlist[5]>0: #定缺
			tmp[5]=1
			valid.append(tmp.copy())
			tmp[5]=2
			valid.append(tmp.copy())
			tmp[5]=3
			valid.append(tmp.copy())
		if actlist[2]>0: #碰牌
			tmp.fill(0)
			tmp[2]=actlist[2]
			valid.append(tmp.copy())
		if actlist[0]>0: #打牌
			tmp.fill(0)
			start=(self.common_info.que[pos]-1)*9+1
			end=start+8
			index=np.argwhere((hand-start)*(hand-end)<=0)
			if index.size!=0 or (actlist[0]-start)*(actlist[0]-end)<=0: #花猪状态
				for i in hand[index]:
					tmp[0]=i
					valid.append(tmp.copy())
				if (actlist[0]-start)*(actlist[0]-end)<=0:
					tmp[0]=actlist[0]
					valid.append(tmp.copy())
			else:
				for i in hand:
					if i==0: continue
					tmp[0]=i
					valid.append(tmp.copy())
				if actlist[0]<=27 and actlist[0]>=1:
					tmp[0]=actlist[0]
					valid.append(tmp.copy())
		if actlist[4]>0: #胡牌
			tmp.fill(0)
			tmp[4]=1
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
		tmp.fill(0)
		#只有不存在出牌和定缺的情况，才可以不采取任何操作
		if actlist[0]==0 and actlist[5]==0:
			tmp.fill(0)
			valid.append(tmp.copy())
		return valid


#   服务器视图
class sichuanmj_server:
	players=None
	common_info=None
	pool=None
	g=None
	master_cnt=0 #庄家
	step=0
	def __init__(self):
		#初始化公共视图
		self.common_info=sichuanmj_public()
		self.players=np.zeros(4,dtype=sichuanmj_client)
		self.pool=None
		self.bonus=None
		self.g=gui.simple_gui()

		#初始化四个玩家
		for i in range(4):
			self.players[i]=sichuanmj_client()
			self.players[i].me=sichuanmj_private()
			self.players[i].common_info=self.common_info
			self.players[i].me.my_position=i

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
		#初始化私人视图
		for i in range(4):
			self.players[i].me.my_hand.fill(0)
			self.players[i].gang_able=[]
			self.players[i].gang_init=False
		self.mopai()
		self.step=0

	#摸n张牌
	def fetch(self,count):
		seed=int(time.time())
		seed+=self.pool.size
		np.random.seed(seed)
		label=np.random.choice(self.pool.size,size=count,replace=False)
		result=self.pool[label]
		self.pool=np.delete(self.pool,label)
		return result

	#摸牌阶段
	def mopai(self):
		for i in range(4):
			hand=self.players[i].me.my_hand
			hand.fill(0)
			hand[0,:13]=self.fetch(13)
			index=hand[0,:].argsort()
			self.players[i].me.my_hand=hand[:,index]
			self.common_info.hand_cnt[i]=13
		self.common_info.pool_cnt=self.pool.size

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
		cnt_wan=np.argwhere((hand-1)*(hand-9)<=0).size
		cnt_tiao=np.argwhere((hand-10)*(hand-18)<=0).size
		cnt_tong=np.argwhere((hand-19)*(hand-27)<=0).size
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
		if self.common_info.pool_cnt==0: self.endset()
		flag_gang,flag_hu=0,0
		self.step+=1
		actlist=np.zeros(6,dtype=np.int)
		pai=self.fetch(1)[0]
		me=self.players[player_cnt]
		#判断是否可以杠或者胡牌
		fan=me.isHu(pai, player_cnt,Gang)
		if me.isGang(pai,player_cnt).size >0: flag_gang=1
		if fan> 0: flag_hu = 1
		allow=[0]
		if flag_hu==1: allow.append(4)
		if flag_gang==1: allow.append(3)
		actlist[0],actlist[3],actlist[4]=pai,flag_gang,flag_hu
		actlist=self.players[player_cnt].act(actlist,self)
		self.validate_actlist(player_cnt,actlist,allow)
		index=np.argwhere(actlist>0)
		if index.size==0: self.error(player_cnt,92)
		if index[0]==3:
			index=np.argwhere(self.common_info.peng[player_cnt,0,:]==pai)
			if index.size==0:   #暗杠
				self.del_hand(player_cnt,pai,3)
				self.add_gang(player_cnt,player_cnt,1,pai)
				self.bonus[player_cnt] += 2 * 3
				for i in range(4):
					if i==player_cnt:
						self.jiesuan(i,6)
					else:
						self.jiesuan(i,-2)
			else :  #补杠
				#抢杠
				rel_max=0
				for i in range(4):
					fan = self.players[i].isHu(pai,i,1)
					if fan>0:
						actlist.fill(0)
						actlist[4]=fan
						self.players[i].act(actlist,self)
						self.validate_actlist(i,actlist,[4])
						if actlist[4]>0:    #胡牌
							self.hupai(i,player_cnt,fan,pai)
							rel_pos=(i-player_cnt)%4
							if rel_pos>rel_max: rel_pos=rel_max
				if rel_max>0:
					self.players[player_cnt].gang_able.remove(pai)
					rel_max=(rel_max+player_cnt)%4
					self.input(self.next_valid_player(rel_max))
				if actlist[3] == pai:   #有钱的补杠
					self.add_gang(player_cnt,player_cnt,2,pai)
					self.del_peng(player_cnt,pai)
					for i in range(4):
						if i==player_cnt:
							self.jiesuan(i,3)
						else:
							self.jiesuan(i,-1)
				else:
					self.add_gang(player_cnt,player_cnt,3,pai)
					self.del_peng(player_cnt,1)
					self.del_hand(player_cnt,pai,1)
					self.add_hand(player_cnt,pai)
			me.gang_able.remove(actlist[3])
			if self.common_info.pool_cnt>0: self.input(player_cnt,1)
		elif index[0]==4:
				self.hupai(player_cnt,player_cnt,fan,pai)
		elif index[0]==0:
				self.add_hand(player_cnt,pai)
				self.del_hand(player_cnt,actlist[0],1)
				#出牌
				self.output(actlist[0],player_cnt, Gang)


	# 打牌-》判断别人是否碰，杠或者胡
	def output(self,pai,player_cnt,Gang=0):
		me=self.players[player_cnt]
		actlist=np.zeros(6,dtype=np.int)
		flaghu=0
		flagpeng=0
		flaggang=0
		# 把能杠的牌打出
		if pai in me.gang_able: me.gang_albe.remove(pai)
		# 放炮
		for i in range(4):
			if i==player_cnt or self.common_info.status[i]==1 : continue
			fan=self.players[i].isHu(pai,i,Gang)
			if fan>0:
				actlist[4]=fan
				actlist=self.players[i].act(actlist,self)
				self.validate_actlist(i,actlist,[4])
				index=np.argwhere(actlist>0)
				if index.size==1:
					if index[0]!=4: self.error(i,98)
					self.hupai(i,player_cnt,fan,pai)
					rel_pos=(i-player_cnt)%4
					if flaghu<rel_pos: flaghu=rel_pos
					print(player_cnt,"放炮给",i)
		if flaghu>0:
			flaghu=(flaghu+player_cnt)%4
			self.input(self.next_valid_player(flaghu))

		# 碰
		for i in range(4):
			if i==player_cnt or self.common_info.status[i]==1: continue
			index=self.players[i].isPeng(pai)
			if index.size>0:
				cur=index[0]
				actlist.fill(0)
				actlist[2]=pai
				actlist=self.players[i].act(actlist,self)
				index = np.argwhere(actlist > 0)
				self.validate_actlist(i,actlist,[2])
				if index.size==1:
					if index[0]!=2: self.error(i,98)
					#碰牌
					self.add_peng(i,pai)
					self.del_hand(i,pai,2)
					actlist.fill(0)
					actlist[0]=99
					actlist=self.players[i].act(actlist,self)
					self.validate_actlist(i,actlist,[0])
					if actlist[0]<=0: self.error(i,93)
					self.del_hand(i,actlist[0],1)
					self.output(actlist[0],i)

		# 杠
		for i in range(4):
			if i==player_cnt or self.common_info.status[i]==1: continue
			index=self.players[i].isGang(pai,player_cnt)
			if index.size>0:
				actlist.fill(0)
				actlist[3]=pai
				actlist=self.players[i].act(actlist,self)
				index = np.argwhere(actlist > 0)
				self.validate_actlist(i,actlist,[3])
				if index.size==1:
					self.add_gang(i,player_cnt,0,pai)
					self.del_hand(i,pai,3)
					#去除杠牌标志
					self.players[i].gang_able.remove(pai)
					# 奖金结算
					self.jiesuan(i,2)
					self.jiesuan(player_cnt,-2)
					rel_pos = (i - player_cnt) % 4
					if flaggang < rel_pos: flaggang = rel_pos
					print(player_cnt,"放杠给",i)
		if flaggang > 0:
			flaggang = (flaggang + player_cnt ) % 4
			self.input(self.next_valid_player(flaggang),Gang)

		#出牌
		#放入堂子中
		self.common_info.drop[0,self.step]=pai
		self.common_info.drop[1, self.step] = player_cnt
		self.input(self.next_valid_player(player_cnt))

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
		peng[:,index[0]].fill(0)
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
			if next==player_cnt: self.error(player_cnt,201)
		return next

	def validate_actlist(self,player_cnt,actlist,allow):
		index=np.argwhere(actlist>0)
		if index.size>1: self.error(player_cnt,90) #命令多于一个
		if index.size==0: return
		if index.size==1 and (not index[0] in allow): self.error(player_cnt,91)

	def jiesuan(self,player,money):
		self.bonus[player]+=money
		#以后完成，每当有钱的输赢产生的时候就喂ai

	def hupai(self,p_hu,p_from,fan,pai):
		self.add_hand(p_hu,pai)
		self.common_info.status[p_hu]=1
		base=2**min(3,fan)  #3番封顶
		if p_hu!=p_from: #放炮
			self.jiesuan(p_hu,base)
			self.jiesuan(p_from, -base)
		else:
			base+=1 #自摸加底
			self.jiesuan(p_hu, base*3)
			for i in range(4):
				if i==p_hu: continue
				self.jiesuan(p_hu, -base)
		#如果三个人都胡牌了，游戏结束
		hu_cnt=np.argwhere(self.common_info.status==1).size
		if hu_cnt==3: self.endset()

	def endset(self):
		#三个人都胡牌了，不用查叫查花猪
		if self.common_info.pool_cnt>0: return
		if np.argwhere(self.common_info.status==1).size>=3: return
		# 检查谁有叫
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
				mfan[i]=maxfan
			else:
				bad.append(i)
		#花猪赔所有
		for i in hua:
			self.jiesuan(i,-3*2**3)
			for j in range(4):
				if i!=j: self.jiesuan(j,2**3)
		#没叫赔有叫
		for i in bad:
			for j in good:
				self.jiesuan(i,-2**max(3,mfan[j]))
				self.jiesuan(j,2**max(3,mfan[j]))
		#没叫返杠钱
		for i in bad:
			gang=self.common_info.gang[i,:,:]
			for j in range(4):
				if gang[0,j]==0: continue
				if gang[2,j]==0: #点杠
					self.jiesuan(i,-2)
					self.jiesuan(gang[3,j],2)
					continue
				if gang[2,j]==1: #暗杠
					self.jiesuan(i,-6)
					for k in range(4):
						if k==i: continue
						self.jiesuan(k,2)
					continue
				if gang[2,j]==2: #补杠
					self.jiesuan(i,-3)
					for k in range(4):
						if k==i: continue
						self.jiesuan(k,1)

