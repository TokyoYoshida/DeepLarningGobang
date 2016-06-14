from __future__ import print_function
MODE_NORMAL = 0
MODE_SPARK = 1
MODE_TENSORFLOW = 2
MODE = MODE_TENSORFLOW #choise mode

TENSORFLOW_TRAINMODE = 1
TENSORFLOW_FIHGTMODE = 2
TENSORFLOW_MODE = TENSORFLOW_FIHGTMODE

TENSORFLOW_MODEL_FILENAME = "model.ckpt"

if MODE == MODE_NORMAL:
	import sys
if MODE == MODE_SPARK:
	from pyspark import SparkContext
import copy
import functools
from random import random
from flask import Flask, Response, request
import os
import urllib
import urllib3
import SimpleHTTPServer,SocketServer
import urlparse

if MODE == MODE_TENSORFLOW:
	import tensorflow as tf
	import numpy

#board stratage part
BOARD_ROW_SIZE = 3
BOARD_COL_SIZE = 3
NUMBER_OF_COMPLETE_CASE = 3

MAX_DEPTH = 5
MAX_DEPTH_OF_TENSORFLOW = MAX_DEPTH*2
MONTE_CARLO_TREE_SEARCH_NUMBER_OF_RANDOM_CHOICE =10

#tensorflow base part
if MODE == MODE_TENSORFLOW:
	board_SIZE = BOARD_ROW_SIZE*BOARD_COL_SIZE
	HIDDEN_UNIT_SIZE = 32
	TRAIN_DATA_SIZE = 90

	raw_input = numpy.loadtxt(open("input.csv"), delimiter=",")
	[decision, board]  = numpy.hsplit(raw_input, [1])

	[decision_train, decision_test] = numpy.vsplit(decision, [TRAIN_DATA_SIZE])
	[board_train, board_test] = numpy.vsplit(board, [TRAIN_DATA_SIZE])

def inference(board_placeholder):
  with tf.name_scope('hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([board_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(board_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 1], stddev=0.1), name="output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias")
    output = tf.matmul(hidden1_output, output_weight) + output_bias
  return output

def loss(output, decision_placeholder, loss_label_placeholder):
  with tf.name_scope('loss') as scope:
    loss = tf.nn.l2_loss(output - decision_placeholder)
    tf.scalar_summary(loss_label_placeholder, loss)
  return loss

def training(loss):
  with tf.name_scope('training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step


#tensorflow init

if MODE == MODE_TENSORFLOW:
	global decision_placeholder,board_placeholder,loss_label_placeholder,\
	   trainDataList,feed_dict_test,output,loss,training_op,saver,\
	   sess,summary_op,summary_writer
	print("tensorflow initialize")
	with tf.Graph().as_default():
		decision_placeholder = tf.placeholder("float", [None, 1], name="decision_placeholder")
		board_placeholder = tf.placeholder("float", [None, board_SIZE], name="board_placeholder")
		loss_label_placeholder = tf.placeholder("string", name="loss_label_placeholder")

		# nextBoardRawData = getNextBoardData()
		# nextBoardFlatData =  map(lambda e: e.val, flat(nextBoardRawData._board))
		# oneTrainBoard =  [nextBoardFlatData]
		# oneTrainTeach = [getTeachData(oneTrainBoard[0])]
		# target_board = [trainBoardDat[0]]

		# feed_dict_train={
		# 	decision_placeholder: oneTrainTeach,
		# 	board_placeholder: oneTrainBoard,
		# 	loss_label_placeholder: "loss_train"
		# }
		#
		output = inference(board_placeholder)
		loss = loss(output, decision_placeholder, loss_label_placeholder)
		training_op = training(loss)

		saver = tf.train.Saver()
		if os.path.exists(TENSORFLOW_MODEL_FILENAME) == False:
			sess =  tf.Session()
			init = tf.initialize_all_variables()
			sess.run(init)
		else:
			print("model restore mode")
			sess = tf.InteractiveSession()
			sess.run(tf.initialize_all_variables())
			saver.restore(sess, TENSORFLOW_MODEL_FILENAME)

		summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)

def decisionByTensorFlow(board):
	# print("decisionByTensorFlow")
	# board.description()
	targetBoardFlatData =  map(lambda e: e.val, flat(board._board))
	feed_dict_decision={
		decision_placeholder: [[0]],
		board_placeholder: [targetBoardFlatData],
		loss_label_placeholder: "loss_decision"
	}
	decision = sess.run(output, feed_dict=feed_dict_decision)
	decisionOfNum =  map(lambda e: int(round(e,0)) ,decision)[0]
	# print(decisionOfNum)
	if 0 <= decisionOfNum and decisionOfNum < BOARD_ROW_SIZE*BOARD_COL_SIZE:
		decisionOfPos =  convert_pos_of_num_to_pos(decisionOfNum)
	else:
		decisionOfPos = None
	# print ("tensorflow decision is")
	# print(decisionOfPos)
	return decisionOfPos

#board part
def flat(input):
	    ret =[]
	    for elem in input:
	        ret += flat(elem) if type(elem) is list else [elem]
	    return ret

class Turn():
		White = 1
		Black = 2

		def __init__(self,val):
			self.val = val

		def getNext(self):
		    if self.val == Turn.White:
		        return Turn(Turn.Black)
		    else:
		        return Turn(Turn.White)

		def getSquareState(self):
		    if self.val == Turn.White:
		        return SqueareState(SqueareState.White)
		    else:
		        return SqueareState(SqueareState.Black)

		def description(self):
		    if self.val == Turn.White:
		        return "O"
		    else:
		        return "X"

class SqueareState():
		Blank = 0
		White = 1
		Black = 2

		def __init__(self,val):
			self.val = val

		def canPut(self):
		    if self.val == SqueareState.Blank:
		        return True
		    else:
		        return False

		def description (self):
		    if self.val == SqueareState.Blank:
		        return "- "
		    elif self.val == SqueareState.White:
		        return "O "
		    elif self.val == SqueareState.Black:
		        return "X "

# class Pos:
#         row = 0
#         col = 0
#
#         def __init__(self, row ,col):
#             self.row = row
#             self.col = col
#
#         def isValid():
#             return (0 <= self.row and self.row < BOARD_ROW_SIZE) and \
#                    (0 <= self.col and self.col < BOARD_COL_SIZE)
#
#         def description(self):
#             return str(self.row) + "," + str(self.col)


class OperationValue:
        row = 0
        col = 0
        def __init__(self, row ,col):
            self.row = row
            self.col = col

class Direction():
		RightUp   = 1
		Right     = 2
		RightDown = 3
		Down      = 4

		def __init__(self,val):
			self.val = val

		def geOperationValue(self):
		    if self.val == Direction.RightUp:
		        return OperationValue(-1 ,1 )
		    elif self.val == Direction.Right:
		        return OperationValue( 0 ,1 )
		    elif self.val == Direction.RightDown:
		        return OperationValue( 1 ,1 )
		    elif self.val == Direction.Down:
		        return OperationValue( 1 ,0 )

class GameResult():
    Undecided = 0
    Win       = 1
    Lose      = 2

class Board:
        def __init__(self, board = None):
            if board is None:
                self._board = [[SqueareState(SqueareState.Blank) for i in range(BOARD_COL_SIZE)] for j in range(BOARD_ROW_SIZE)]
            else:
                self._board = copy.deepcopy(board._board)

        def get(self,pos):
            return self._board[pos[0]][pos[1]]

        def set(self,pos,newValue):
			# print(pos[0],pos[1],self._board[pos[0]][pos[1]].val)
			assert self._board[pos[0]][pos[1]].val == SqueareState.Blank ,str(pos[0])+" "+str(pos[1])+"/"+str(self._board[pos[0]][pos[1]].val)
			self._board[pos[0]][pos[1]].val = newValue

        def canPut(self,pos):
            if self.get(pos).val == SqueareState.Blank:
                return True
            else:
                return False

        def description(self):
            for r in self._board:
                for c in r:
                    print("%s " % c.description(),end="")
                print("")

        def countHits(self, pos, targetState, direction):
            # print(pos.description())
            if pos[0] == 0   and                                                        direction.val == Direction.RightUp:
                return 0
            elif                                     pos[1] == BOARD_COL_SIZE and direction.val == Direction.RightUp:
                return 0
            elif                                     pos[1] == BOARD_COL_SIZE and direction.val == Direction.Right  :
                return 0
            elif                                     pos[1] == BOARD_COL_SIZE and direction.val == Direction.RightDown  :
                return 0
            elif pos[0] == BOARD_ROW_SIZE and                                     direction.val == Direction.RightDown  :
                return 0
            elif pos[0] == BOARD_ROW_SIZE and                                     direction.val == Direction.Down  :
                return 0
            elif self.get(pos).val == targetState.val:
                return 1 + self.countHits(\
                                     	(pos[0] + direction.geOperationValue().row,\
                                         pos[1] + direction.geOperationValue().col),
                                     targetState,\
                                     direction)
            else:
                return 0

        def generatePosList(self):
            pos_list = [[(0,0) for i in range(BOARD_COL_SIZE)] for j in range(BOARD_ROW_SIZE)]
            ri = 0
            for r in self._board:
                ci = 0
                for c in r:
                    pos_list[ri][ci] = (ri,ci)
                    ci = ci + 1
                ri = ri + 1
            return pos_list

        def countComplete(self, e, targetState, direction):
            if self.countHits(e, targetState, direction) == NUMBER_OF_COMPLETE_CASE:
                return 1
            else:
                return 0

        def mapPosListCol(self, e,  targetState, direction):
            return functools.reduce(lambda a,b:a+b,list(map(functools.partial(self.countComplete, targetState=targetState, direction=direction),e)))

        def countAllCompletes(self,targetState):
			l = self.generatePosList()
			countList = \
			    list(map(functools.partial(self.mapPosListCol, targetState=targetState, direction=Direction(Direction.Right)),l)) +\
			    list(map(functools.partial(self.mapPosListCol, targetState=targetState, direction=Direction(Direction.RightUp)),l)) + \
			    list(map(functools.partial(self.mapPosListCol, targetState=targetState, direction=Direction(Direction.RightDown)),l)) +\
			    list(map(functools.partial(self.mapPosListCol, targetState=targetState, direction=Direction(Direction.Down)),l))

			# print(countList)
			return functools.reduce(lambda a,b:a+b,countList)

        def evaluate(self, targetTurn):
            if self.countAllCompletes(targetTurn.getNext().getSquareState()) > 0:
                return GameResult.Lose
            elif self.countAllCompletes(targetTurn.getSquareState()) > 0:
                return GameResult.Win
            else:
                return GameResult.Undecided

		def toFlatList(self):
			return map(lambda e: e.val, flat(self._board))

		def toReverse(self):
			newBoard = Board(self)
			for ri in len(newBoard._board):
				for ci in len(newBoard._board[0]):
					e = newBoard._board[ri][ci].val
					if e == SqueareState.White:
						newBoard._board[ri][ci].val = SqueareState.Blank
					elif e == SqueareState.Blank:
						newBoard._board[ri][ci].val = SqueareState.White



# class TryResult:
#         point = 0
#         pos = (0,0)
#
#         def __init__(self, point ,pos):
#             self.point = point
#             self.pos = pos

def tryNext(pos, board, turn, depth ,maxDepth = MAX_DEPTH):
    _board = Board(board)
    # print(pos.description(),turn.description())
    _board.set(pos, turn.getSquareState().val)

    evalResult = _board.evaluate(turn)

    if evalResult == GameResult.Lose:
        return (-1+depth-maxDepth, pos)
    elif evalResult == GameResult.Win:
        # print(_board.description())
        return (maxDepth-depth, pos)
    else:
        if depth == maxDepth:
            return (0 ,pos)
        else:
            if depth == maxDepth:
                return (0, pos)
            else:
                return decisionNext(_board, turn.getNext() ,depth)

def choiceNextMoveCandidates(board,length):
	listAll = flat(board.generatePosList())
	listCanPut = flat(list(filter(functools.partial(board.canPut), listAll)))
	# print(listCanPut)
	return randomChoice(listCanPut,length)

# def decisionNext_local(board, turn ,depth):
# 	listChoiced = choiceNextMoveCandidates(board,MONTE_CARLO_TREE_SEARCH_NUMBER_OF_RANDOM_CHOICE)
# 	if len(listChoiced) == 0:
# 		return (0 ,(-1,-1))
# 	return sorted(tryResult, key=lambda tryresult: tryresult[0],reverse=True)[0]
#

def decisionNext( board, turn ,depth ,maxDepth = MAX_DEPTH ,localMode = True):
	listChoiced = choiceNextMoveCandidates(board,MONTE_CARLO_TREE_SEARCH_NUMBER_OF_RANDOM_CHOICE)
	if len(listChoiced) == 0:
		return (0 ,(-1,-1))
	if localMode == True:
		tryResult = flat(list(map( functools.partial(tryNext, board=board, turn=turn ,depth=depth+1,maxDepth=MAX_DEPTH), listChoiced)))
	else:
		tryResultNonFlat = sparkContext.parallelize( listChoiced ,partiTions).map( lambda p: tryNext(p ,board, turn ,depth+1, MAX_DEPTH))
		tryResult = flat(tryResultNonFlat.collect())

	if MODE == MODE_TENSORFLOW and TENSORFLOW_MODE == TENSORFLOW_FIHGTMODE:
		if turn == Turn.Black:
			tb = board.toReverse()
		else:
			tb = board
		posOfTensor = decisionByTensorFlow(tb)
		if posOfTensor is not None:
			if board.canPut(posOfTensor):
				tryResultNonFlatOfTensorFlow = flat(list(map( functools.partial(tryNext, board=board, turn=turn ,depth=depth+1 ,maxDepth=MAX_DEPTH_OF_TENSORFLOW), [posOfTensor])))
				tryResultOfTensorFlow = flat(tryResultNonFlatOfTensorFlow)
				tryResult = tryResult + tryResultOfTensorFlow

	if depth == 0:
		print(tryResult)

	return sorted(tryResult, key=lambda tryresult: tryresult[0],reverse=True)[0]

def decisionCanput(board):
	print("decisionCanput")
	listAll = flat(board.generatePosList())
	listCanPut = flat(list(filter(functools.partial(board.canPut), listAll)))
	print(listCanPut)
	if len(listCanPut) == 0:
		return (-1,-1)
	return listCanPut[0]

def randomChoice(targetList,length):
	zeroList = [0]*len(targetList)
	randList = map( lambda _: int(random()*len(targetList)) , zeroList)
	return map(lambda e:e[1], sorted(zip(randList, targetList) ,key=lambda e: e[0]))[:length]

#web server and client part
PORT = 4000
URL =  'http://localhost:'+str(PORT)
global_board = Board()

# app = Flask(__name__, static_url_path='', static_folder='public')
# app.add_url_rule('/', 'root', lambda: app.send_static_file('index.html'))
#
# @app.route('/board', methods=['GET'])
# def server():
#     if request.method == 'GET':
# 		pos_of_num = request.args.get('pos')
# 		if pos_of_num is not None:
# 		    res = str(serve_pos(int(pos_of_num)))
# 		board_of_flat_list = request.args.get('board')
# 		if board_of_flat_list is not None:
# 		    res = serve_board(board_of_flat_list)
# 		print("return = ")
# 		print(res)
# 		return Response(res, mimetype='text/html', headers={'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*'})

def request_board(board_of_flat_list):
    http = urllib3.PoolManager()
    r = http.request('GET',URL + "?board=" + request_board)
    return r.data

def request_pos(pos_of_num):
    http = urllib3.PoolManager()
    r = http.request('GET',URL + "?pos=" + pos_of_num)
    return r.data

# global_board = None
def serve_board(board_of_flat_list):
	global_board.__init__(Board(convert_board_of_flat_list_to_board(board_of_flat_list)))
	return "1"

def convert_board_of_flat_list_to_board(board_of_flat_list):
	# print("board_of_flat_list")
	# print(board_of_flat_list)
	board = Board()
	index = 0
	for r in range(BOARD_ROW_SIZE):
		for c in range(BOARD_COL_SIZE):
			# print(int(board_of_flat_list[index]))
			board.set((r,c) ,int(board_of_flat_list[index]))
			index = index + 1
	return board

def serve_pos(put_of_num):
	print("ok1")
	print(put_of_num)
	pos = convert_pos_of_num_to_pos(put_of_num)
	global_board.description()
	global_board.set(pos,SqueareState.Black)
	print("ok2")
	global_board.description()
	if MODE == MODE_SPARK:
		tryResult = decisionNext(global_board, Turn(Turn.White), 0 ,localMode = False)
	else:
		tryResult = decisionNext(global_board, Turn(Turn.White), 0)
	# print(tryResult[0])
	# print(tryResult[1][0])
	decisionPos = tryResult[1] if tryResult[1][0] >= 0 else decisionCanput(global_board)
	if decisionPos[0] != -1:
		global_board.set(decisionPos, SqueareState.White)
	global_board.description()
	print("ok3")
	print(tryResult)
	return convert_pos_to_pos_of_num(decisionPos)

def convert_pos_of_num_to_pos(pos_of_num):
	# print("pos c1")
	# print(pos_of_num)
	posRow = pos_of_num / BOARD_COL_SIZE
	posCol = pos_of_num % BOARD_COL_SIZE
	# print("pos c2")
	return (posRow,posCol)

def convert_pos_to_pos_of_num(pos):
	# print("pos cc1")
	# print(pos)
	pos_of_num = pos[0]*BOARD_COL_SIZE + pos[1]
	# print("pos cc2")
	return pos_of_num


class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
   		parsedParams = urlparse.urlparse(self.path)
   		queryParsed = urlparse.parse_qs(parsedParams.query)
		for q in queryParsed:
			if q == 'pos':
				pos_of_num = int(queryParsed['pos'][0])
	 			res = str(serve_pos(int(pos_of_num)))
			if q == 'board':
				board_of_str = queryParsed['board'][0]
				board_of_flat_list = board_of_str.split(',')
				res = serve_board(board_of_flat_list)
				print(global_board.description())
		self.send_response(200)
		self.send_header('Content-type', 'text/html; charset=utf-8')
		self.send_header('Content-length', len(res))
		self.end_headers()
		self.wfile.write(res)

if MODE == MODE_SPARK:
	sparkContext = SparkContext(appName="gobang")
	partiTions = 1

if TENSORFLOW_MODE != TENSORFLOW_TRAINMODE:
	host = 'localhost'
	port = 4000
	httpd = SocketServer.TCPServer(("", port), MyHandler)
	print('serving at port', port)
	httpd.serve_forever()

# if __name__ == '__main__':
#     app.run(port=int(os.environ.get("PORT",PORT)),threaded=True)

# #entry point
# # print("start--------------")
# # sparkContext = SparkContext(appName="gobang")
# # partiTions = 1
# # global_board.set((0,0),SqueareState.White)
# # global_board.set((1,1),SqueareState.Black)
# # global_board.set((0,2),SqueareState.White)
# # global_board.set((0,1),SqueareState.Black)
# # r = decisionNext_local(global_board, Turn(Turn.White), 0)
# # print("result")
# # print(r)
#
# # while True:
# # 	userPos = raw_input("pos: ")
# # 	print(serve_pos(int(userPos)))

#tensorflow train part
trainBoardDat =\
    [[1,2,1,\
      0,2,0,\
      0,0,0],\
    [2,2,0,\
      0,1,0,\
      0,0,1],\
    [1,2,0,\
      1,0,2,\
      0,0,0],\
    [2,2,0,\
      1,1,0,\
      0,0,0]]

trainTeachDat = [[7],[2],[6],[5]]

testBoardDat =\
    [[1,2,1,\
      0,2,0,\
      0,0,0],\
    [2,2,0,\
      0,1,0,\
      0,0,1],\
    [1,2,0,\
      1,0,2,\
      0,0,0],\
    [2,2,0,\
      1,1,0,\
      0,0,0]]
testTeachDat = [[7],[2],[6],[5]]

def getNextBoardData():
	board = Board()
	index = 0
	rMax = int(random()*BOARD_ROW_SIZE*BOARD_COL_SIZE-1)
	firstFlg = True
	if (rMax % 2) == 0:
		state = SqueareState.White
	else:
		state = SqueareState.Black

	listChoiced = choiceNextMoveCandidates(board,rMax)
	for pos in listChoiced:
		board.set(pos ,state)
		if state == SqueareState.Black:
			state = SqueareState.White
		else:
			state = SqueareState.Black

	return board

def getTeachData(board_of_flat_list):
	# print(board_of_flat_list)
	_board = convert_board_of_flat_list_to_board(board_of_flat_list)
	tryResult = decisionNext(_board, Turn(Turn.White), 0)
	decisionPos = tryResult[1] if tryResult[1][0] >= 0 else decisionCanput(_board)
	return decisionPos

def makeTrainDataList(makeNum):
	boardList = [None]*makeNum
	teachList = [None]*makeNum
	for i in range(makeNum):
		print("make train data")
		nextBoardRawData = getNextBoardData()
		nextBoardRawData.description()
		nextBoardFlatData =  map(lambda e: e.val, flat(nextBoardRawData._board))
		boardList[i] =  nextBoardFlatData
		teachDataRaw = getTeachData(nextBoardFlatData)
		teachList[i] = [convert_pos_to_pos_of_num(teachDataRaw)]
		print(teachDataRaw)
	return (teachList,boardList)




def TrainStart():
		trainDataList = makeTrainDataList(10) # make training data number

		for rep in range(1):
			for data_index in range(1):
				oneTrainTeach = trainDataList[0]
				oneTrainBoard = trainDataList[1]

				# oneTrainTeach = [trainTeachDat[data_index]]
				# oneTrainBoard = [trainBoardDat[data_index]]

				# nextBoardRawData = getNextBoardData()
				# nextBoardRawData.description()
				# nextBoardFlatData =  map(lambda e: e.val, flat(nextBoardRawData._board))
				# oneTrainBoard =  [nextBoardFlatData]
				# oneTrainTeach = [[convert_pos_to_pos_of_num(getTeachData(oneTrainBoard[0]))]]
				# print(oneTrainBoard)
				# print("oneTrainTeach")
				# print(oneTrainTeach)
				feed_dict_train={
					decision_placeholder: oneTrainTeach,
					board_placeholder: oneTrainBoard,
					loss_label_placeholder: "loss_train"
				}


				for step in range(10000): # training number
				#   print(feed_dict_train)
				  sess.run(training_op, feed_dict=feed_dict_train)
				  if step % 100 == 0:
				    # summary_str = sess.run(summary_op, feed_dict=feed_dict_test)
				    # print(sess,summary_op,feed_dict_train)
				    summary_str = sess.run(summary_op, feed_dict=feed_dict_train)
				    summary_writer.add_summary(summary_str, step)

		feed_dict_test={
			decision_placeholder: testTeachDat,
			board_placeholder: testBoardDat,
			loss_label_placeholder: "loss_test"
		}

		loss_test = sess.run(loss, feed_dict=feed_dict_test)
		best_match = sess.run(output, feed_dict=feed_dict_test)
		print ("test result")
		print (sess.run(tf.nn.l2_normalize(decision_placeholder, 0), feed_dict=feed_dict_test))
		print (map(lambda e: int(round(e,0)) ,best_match))
		save_path = saver.save(sess, TENSORFLOW_MODEL_FILENAME)


if MODE == MODE_TENSORFLOW and TENSORFLOW_MODE == TENSORFLOW_TRAINMODE:
	TrainStart()

	targetBoardRawData = getNextBoardData()
	targetBoardRawData.description()
	#targetBoardFlatData =  map(lambda e: e.val, flat(targetBoardRawData._board))
	#targetBoard = [targetBoardFlatData]
	decisionOfPos = decisionByTensorFlow(targetBoardRawData)
	print ("tensorflow decision is")
	print(decisionOfPos)
