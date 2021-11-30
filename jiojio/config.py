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
    available_models = ["default"]
    models_with_dict = ["medicine", "tourism"]

    def __init__(self):

        temp_dir = '/home/ubuntu/github/jiojio/train_dir'
        # main setting
        self.jiojio_home = os.path.expanduser('~/.jiojio')
        self.trainFile = os.path.join("data", "small_training.utf8")
        self.testFile = os.path.join("data", "small_test.utf8")
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.homepath = self._tmp_dir.name
        # self.temp_dir = os.path.join(self.homepath, ".pkuseg", "temp")
        self.temp_dir = os.path.join(temp_dir, 'temp')
        self.readFile = os.path.join("data", "small_test.utf8")
        self.outputFile = os.path.join("data", "small_test_output.utf8")

        self.modelOptimizer = "crf.adf"
        self.rate0 = 0.05  # 梯度初始值
        self.rate1 = 0.6  # 梯度的下降率
        # self.reg = 1
        # self.regs = [1]
        # self.regList = self.regs.copy()
        # 0 for 0-initialization of model weights, 1 for random init of model weights
        self.random = 1
        # tok.acc (token accuracy), str.acc (string accuracy), f1 (F1-score)
        self.evalMetric = ("f1")
        self.trainSizeScale = 1  # for scaling the size of training data
        self.ttlIter = 20  # of training iterations
        # self.nUpdate = 10  # for ADF training，样本分 10 份
        self.outFolder = os.path.join(self.temp_dir, "output")
        self.save = 1  # save model file
        self.rawResWrite = True
        self.miniBatch = 99  # mini-batch in stochastic training
        self.nThread = 10  # number of processes
        # ADF training
        self.upper = 0.995  # was tuned for nUpdate = 10
        self.lower = 0.6  # was tuned for nUpdate = 10

        # global variables
        self.metric = 'f1'
        self.regularization = 1
        self.outDir = self.outFolder
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
        self.fResSum = "summarizeResult.txt"
        self.fResRaw = "rawResult.txt"
        self.fOutput = "outputTag-{}.txt"

        self.fFeatureTrain = os.path.join(self.temp_dir, "ftrain.txt")
        self.fGoldTrain = os.path.join(self.temp_dir, "gtrain.txt")
        self.fFeatureTest = os.path.join(self.temp_dir, "ftest.txt")
        self.fGoldTest = os.path.join(self.temp_dir, "gtest.txt")

        self.modelDir = os.path.join(temp_dir, "models", "ctb8")

        self.fModel = os.path.join(self.modelDir, "model.txt")

        # start and end token
        self.start_token = '[START]'
        self.end_token = '[END]'

        # feature
        self.numLetterNorm = True  # 将所有的 数字、字母，正规化，即用统一字符替代
        self.featureTrim = 0  # 特征出现频次过低则丢弃，当数据量超大时使用
        self.wordFeature = True  # 需要返回 词汇 特征
        self.wordMax = 6
        self.wordMin = 2
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

        assert self.rate0 > 0
        assert self.trainSizeScale > 0
        assert self.ttlIter > 0
        assert self.miniBatch > 0
        assert self.regularization > 0


config = Config()
