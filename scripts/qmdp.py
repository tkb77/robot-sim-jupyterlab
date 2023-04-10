import sys
sys.path.append('../scripts/')
from dp_policy_agent import *
from dynamic_programming import *

class QmdpAgent(DpPolicyAgent):
    def __init__(self, time_interval, estimator, goal, puddles, sampling_num=10, widths=np.array([0.2, 0.2, math.pi/18]).T, \
                 puddle_coef=100.0, lowerleft=np.array([-4, -4]).T, upperright=np.array([4, 4]).T): 
        super().__init__(time_interval, estimator, goal, puddle_coef, widths, lowerleft, upperright)
        
        self.dp = DynamicProgramming(widths, goal, puddles, time_interval, sampling_num) #DPのオブジェクトを持たせる
        self.dp.value_function = self.init_value()
        self.evaluations = np.array([0.0, 0.0, 0.0]) #Q_MDP値を入れる。描画用
        self.current_value = 0.0 #ファイルから読み込んで価値関数をセット
        self.history = [(0, 0)] #行動の履歴を記録
        
    def init_value(self):
        tmp = np.zeros(self.dp.index_nums)
        for line in open("value.txt", "r"):
            d = line.split()
            tmp[int(d[0]), int(d[1]), int(d[2])] = float(d[3])
            
        return tmp
    
    def evaluation(self, action, indexes):
        return sum([self.dp.action_value(action, i, out_penalty=False) for i in indexes])/len(indexes) #パーティクルの重みの正規化が前提
        
    def policy(self, pose, goal=None):
        indexes = [self.to_index(p.pose, self.pose_min, self.index_nums, self.widths) for p in self.estimator.particles]
        self.current_value = sum([self.dp.value_function[i] for i in indexes])/len(indexes) #描画用に計算
        self.evaluations = [self.evaluation(a, indexes) for a in self.dp.actions]
        self.history.append(self.dp.actions[np.argmax(self.evaluations)])
        
        if self.history[-1][0] + self.history[-2][0] == 0.0 and self.history[-1][1] + self.history[-2][1] == 0.0: #2回の行動で停止していたら前進
            return (1.0, 0.0)
        
        return self.history[-1]
    
    def draw(self, ax, elems):
        super().draw(ax, elems)
        elems.append(ax.text(-4.5, -4.6, "{:.3} => [{:.3}, {:.3}, {:.3}]".format(self.current_value, *self.evaluations), fontsize=8))
