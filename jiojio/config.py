import os
import tempfile


class Config:
    lineEnd = "\n"
    biLineEnd = "\n\n"
    triLineEnd = "\n\n\n"
    undrln = "_"
    blank = " "
    tab = "\t"
    star = "*"
    slash = "/"
    comma = ","
    delimInFeature = "."
    B = "B"
    num = "0123456789.几二三四五六七八九十千万亿兆零１２３４５６７８９０％"
    letter = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ／・－"
    mark = "*"
    model_urls = {
        "postag": "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip",
    }
    model_hash = {
        "postag": "afdf15f4e39bc47a39be4c37e3761b0c8f6ad1783f3cd3aff52984aebc0a1da9",
    }

    def __init__(self):

        self.train_dir = '/home/cuichengyu/github/jiojio/train_dir'
        # main setting
        self.jiojio_home = os.path.expanduser('~/.jiojio')
        # self.trainFile = os.path.join("data", "small_training.utf8")
        # self.testFile = os.path.join("data", "small_test.utf8")
        self.temp_dir = os.path.join(self.train_dir, 'temp')
        self.readFile = os.path.join("data", "small_test.utf8")
        self.outputFile = os.path.join("data", "small_test_output.utf8")

        self.modelOptimizer = "crf.adf"
        self.initial_learning_rate = 0.05  # 梯度初始值
        self.dropping_rate = 0.94  # 维持学习速率，越大则下降越快(0~1)

        # 0 for 0-initialization of model weights, 1 for random init of model weights
        self.random = 1
        # tok.acc (token accuracy), str.acc (string accuracy), f1 (F1-score)
        self.evalMetric = "f1"
        self.trainSizeScale = 1  # for scaling the size of training data
        self.train_epoch = 20  # of training iterations
        # self.nUpdate = 10  # for ADF training，样本分 10 份
        self.outFolder = os.path.join(self.temp_dir, "output")
        self.save = 1  # save model file
        self.rawResWrite = True
        self.miniBatch = 2000  # mini-batch in stochastic training
        self.nThread = 10  # number of processes
        # ADF training
        self.upper = 0.995  # was tuned for nUpdate = 10
        self.lower = 0.6  # was tuned for nUpdate = 10

        # global variables
        self.metric = 'f1'
        self.regularization = 2
        self.testrawDir = "rawinputs/"
        self.testinputDir = "inputs/"
        self.testoutputDir = "entityoutputs/"

        # self.GL_init = True
        self.weightRegMode = "L2"  # choosing weight regularizer: L2, L1)

        self.c_train = os.path.join(self.temp_dir, "train.conll.txt")
        self.f_train = os.path.join(self.temp_dir, "train.feat.txt")

        self.c_test = os.path.join(self.temp_dir, "test.conll.txt")
        self.f_test = os.path.join(self.temp_dir, "test.feat.txt")

        self.fTune = "tune.txt"
        self.fLog = "trainLog.txt"
        self.fResSum = "summarize_result.txt"
        self.fResRaw = "raw_result.txt"
        self.fOutput = "outputTag-{}.txt"

        self.fFeatureTrain = os.path.join(self.temp_dir, "ftrain.txt")
        self.fGoldTrain = os.path.join(self.temp_dir, "gtrain.txt")
        self.fFeatureTest = os.path.join(self.temp_dir, "ftest.txt")
        self.fGoldTest = os.path.join(self.temp_dir, "gtest.txt")

        self.model_dir = os.path.join(os.path.dirname(self.train_dir),
                                      "models/default_model")
        self.available_models = ['default']
        self.models_with_dict = ['default']
        # start and end token
        self.start_token = '[START]'
        self.end_token = '[END]'

        # feature
        self.norm_text = False  # 将所有的 数字、字母，正规化，即用统一字符替代
        self.feature_trim = 5  # 特征出现频次过低则丢弃，当数据量超大时使用
        self.wordFeature = True  # 需要返回 词汇 特征
        self.word_max = 5
        self.word_min = 2
        self.nLabel = 2
        self.order = 1

    def globalCheck(self):
        if self.evalMetric == "f1":
            self.metric = "f-score"
        elif self.evalMetric == "tok.acc":
            self.metric = "token-accuracy"
        elif self.evalMetric == "str.acc":
            self.metric = "string-accuracy"
        else:
            raise Exception("invalid eval metric")

        assert self.initial_learning_rate > 0
        assert self.trainSizeScale > 0
        assert self.train_epoch > 0
        assert self.miniBatch > 0
        assert self.regularization > 0


config = Config()
config.globalCheck()
