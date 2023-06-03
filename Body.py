import numpy as np

# Матрицы весов и смещения, полученные при обучении
BodyW1 = np.array([[ 7.22287952e-01, 4.49208832e-01, 5.91902688e-01, 5.37015087e-01,
  -2.21649908e-01, 1.93263169e-01,-1.90717146e+00, 2.45713399e-01,
  -5.32957981e-01, 6.76500781e-01, 6.86155559e-01, 3.25958138e-01,
   9.26335741e-01,-1.12770855e+00, 3.44846033e-01, 5.63726545e-01,
   3.82457882e-01,-1.48470310e-01, 4.41972990e+00, 1.99044145e+00,
  -3.12972523e-01, 3.39765939e-01, 4.24931814e-01, 3.83261800e-01,
   5.06313460e-01],
 [ 6.76492833e-01, 5.42652574e-01, 8.49460063e-01, 2.46331334e-01,
   2.92768203e-02,-7.08469242e-01, 2.20312282e+00, 2.22743088e-01,
  -9.44275370e-05, 4.93466859e-02, 7.20654567e-01, 5.47821870e-02,
   1.57681714e-01, 1.85265358e+00, 3.60314747e-02, 6.13530958e-01,
   9.83382345e-02, 1.71340921e-01, 1.79142570e-01,-2.57364590e-01,
  -2.84393857e-02,-1.17470112e-01, 5.69018417e-02, 4.26096962e-01,
   1.42619964e-01],
 [ 2.35141306e-01, 8.41542028e-01, 1.06523941e+00, 6.38618878e-02,
   3.20963317e+00, 7.75328684e-01, 3.13137505e-02, 5.86041476e-01,
   5.19134483e+00, 5.22842052e-01, 7.38800721e-01, 5.03552078e-01,
   1.20263047e+00,-1.86128471e-01, 9.45158490e-01,-7.57755460e-02,
   9.14265318e-01, 5.63513590e-01, 9.40537613e-01, 7.17981517e-01,
   3.11473539e+00, 1.44142336e-01, 8.24012321e-01, 8.59758373e-01,
   2.78624549e-03],
 [ 8.85412608e-01,-4.45248941e-01,-2.67807038e-01,-9.64296559e-01,
   4.94447523e-02, 2.59566322e+00, 2.79638422e-01,-1.31354494e+00,
   4.50255540e-03, 1.11632549e+00, 2.08540292e+00, 2.69043902e-01,
   6.28730255e-01, 1.63774102e+00, 1.03967553e-01, 1.44312031e-01,
   1.13488756e+00, 1.01863952e+00,-3.90848713e-01, 4.24614960e-01,
   8.89116416e-01,-5.42230895e-02,-1.21789260e+00,-7.44810635e-01,
  -1.99456447e-02]])

Bodyb1 = np.array([[ 3.01248727, 0.17821646,-3.74525576, 6.81894981, 0.76356253, 2.9028599,
  -1.3351147,  0.44034794, 0.14557071, 2.72168149,-0.86340766, 3.2927503,
  -1.18274207,-0.2932924,  4.83782235, 8.73104535,-1.19395304, 6.53119133,
  -3.17768511,-3.86123642, 3.55390808, 9.12819475, 1.84060601,-2.13505979,
   8.30476291]])

BodyW2 = np.array([[-1.1040776,  0.29892917, 0.06429768, 0.94380095, 0.37211304, 0.42508464,
   0.71784861, 1.16968276, 1.08295181, 1.27056321, 2.05116239, 2.31319209,
  -0.41523586,-0.60174144,-1.23761696],
 [ 0.88321739, 0.50094235, 1.18954408, 0.61939882, 1.30361491, 0.85674961,
   0.25268969, 0.27961901, 0.49332352, 0.64213618, 1.07062984, 1.11660501,
  -0.02087645,-0.17851012,-0.59409986],
 [ 1.87057295, 1.20967908, 2.76697365, 1.1917808,  1.40717838, 0.60113075,
   0.27869479, 0.08205341,-0.66799474,-0.64128308,-0.85636373,-0.9842711,
  -0.32593999,-0.38587092, 0.92842229],
 [-0.57471284,-2.06820209,-1.66899566,-0.60362945,-0.92594251, 0.17708653,
   1.06085967, 1.65886424, 2.51232039, 2.50454184, 2.98956285, 3.12131016,
   1.73492175,-1.26747296,-1.7238664,],
 [ 0.90774566, 0.54728399, 0.98358956,-2.02624432, 1.17832926, 0.75786248,
   0.17723901, 0.89688773, 0.84010829,-0.11969322, 0.2811738,  0.6397928,
   0.6795966,  1.33627591, 1.40966754],
 [-2.58244977,-0.61385558, 1.55572862, 0.68860214, 0.4950042,  0.82539007,
   0.71790748, 1.1689409,  0.81674429, 0.43400131, 1.37499224, 1.22065027,
   1.00113395, 1.04169783, 1.18657798],
 [ 0.08128297,-0.11503661, 1.02629088, 0.65185049, 0.73970064, 0.0443336,
  -0.68675086,-0.13659942,-1.12692771,-0.42366448, 0.31245193, 0.11847907,
   0.81869669, 2.48439247, 2.21193132],
 [ 1.56471209, 0.1620935,  0.97633019, 0.37752264, 1.30472171, 0.58815654,
   0.71938846, 0.53654201, 1.07355431, 1.05772547, 0.75544035, 1.00738697,
   0.11932784, 0.15873151,-0.29896172],
 [-0.42347713,-0.7881968, -0.66393625, 3.82778183,-1.37831978,-0.20456437,
  -0.06369982,-0.30077583,-0.30308873, 0.4342797,  0.11103861, 0.1869427,
   1.17211112, 1.56301195, 2.40959588],
 [-0.82308822, 0.21823014,-0.49199198, 0.49195829, 0.46028116, 0.93917296,
   0.91068838, 0.98907198, 1.22553882, 1.05479748, 1.71901345, 1.44277655,
  -1.4553741, -0.03411053, 0.50365783],
 [-0.53699058, 2.12416461, 1.52761581, 1.40039343, 1.35105391, 0.96344642,
   0.64931727, 0.25284785, 0.18653596, 0.30877466, 0.28236227,-0.06812866,
  -0.39630493,-0.49695936,-0.74247559],
 [-0.3321008, -0.24303483,-0.52080643, 0.22396697, 0.19663831, 0.35223064,
   0.82458456, 0.83339126, 1.40170707, 1.41632373, 2.46412467, 1.79142165,
   1.31541708,-0.30260287,-0.35705051],
 [ 1.06044401, 1.57361582, 1.43429955, 1.09755237, 1.77951597, 0.83904936,
   1.01205618, 0.2510419,  0.00704633, 0.03658056,-0.05573575,-0.3132345,
  -1.0566866, -0.20264725, 0.18265385],
 [-0.41983886,-0.52124576,-0.48039856,-0.03965921,-0.07904861, 0.80948209,
   0.76400712, 0.87675458, 1.37935142, 0.86310473,-0.43041322,-0.07040918,
   0.91107221, 0.51767164, 2.52007407],
 [-0.93767565,-1.40934396,-1.0266001, -0.58760258,-0.11344422, 0.43330412,
   1.63277718, 1.48507534, 1.77472137, 1.28501081, 2.11281351, 2.3137055,
   1.72941942, 0.15137343,-1.25595134],
 [-2.2830271, -2.00843628,-2.25833341,-0.16168343,-0.89023668, 0.4469695,
   1.32682274, 1.76280926, 2.7707611,  3.26426151, 3.32568271, 3.43440058,
   2.68093427, 0.21094294,-3.79409198],
 [ 0.1467505,  1.40612574, 1.22877474, 1.16455202, 1.26223273, 0.64647365,
   0.58793465, 0.53721986, 0.30258653, 0.34672297,-0.1282676, -0.0376552,
   0.28057625, 0.28774474, 0.33298585],
 [-2.47468481,-1.29198781,-1.68344414,-0.62347211,-0.41358348, 0.01471781,
   1.34628294, 1.55940625, 2.31350979, 2.09519678, 2.91187654, 2.37981159,
   2.35073118, 1.25597861,-1.17022211],
 [ 3.93358302, 3.46513383, 3.14777894, 1.94102652, 1.71203359, 1.29193472,
   0.40046147, 0.28774684,-1.33949616,-1.28582441,-2.5503922, -5.25126249,
   0.04796791, 0.44606791, 0.17389242],
 [ 2.11625289, 2.11934714, 2.28831402, 1.94610268, 1.98718378, 1.3509143,
   0.31093474, 0.25335372,-0.88037914,-0.72390869,-1.86349013,-1.42678283,
   0.12742812, 0.20624534, 0.07866933],
 [-0.0538763,  0.53872877,-1.47084209,-2.63644018,-0.39480477,-0.17694198,
   0.38686269, 0.50359599, 1.32191014, 1.2471551,  1.21733848, 1.71569754,
   1.89754669, 1.57689375, 1.13395741],
 [-2.18283966,-2.43873738,-3.04956867,-1.34166265,-0.77695254, 0.01786156,
   1.68107369, 1.56570014, 3.27433518, 2.65969322, 3.48683308, 3.43705256,
   2.58246268, 0.89755962,-2.72716483],
 [ 1.33814055,-0.5432383,  0.21173394, 0.23728789, 0.87451606, 0.48931453,
   0.76512235, 0.67591679, 1.35500407, 0.56171732, 1.66456062, 1.03559474,
  -0.78199795,-0.87362039, 0.03748428],
 [ 1.43590484, 0.44487719, 1.83648996, 0.8860207,  0.93942898, 1.16914075,
   0.26674351, 0.46612652,-0.37147218, 0.02711767,-0.66449568, 0.1119694,
   0.42115707, 0.68812507, 0.22748996],
 [-2.00197567,-2.18438154,-2.27081456,-0.80344508,-0.95841648, 0.33833416,
   1.33282601, 1.34292809, 3.11660308, 2.83769819, 3.01611325, 3.37153472,
   2.72578514, 0.28623847,-2.9252311,]])

Bodyb2 = np.array([[-0.27783009, -0.15086772, -0.44259925,  0.67937846,  0.5104666,  -0.06085595,
   1.22997293,  0.74493774,  1.32257813,  1.15441389,  1.90294989,  2.26281422,
   1.20876226,  0.3940299,  -0.7679987 ]])

BodyId = [
    [2], [3], [1], [4], [7], [6], [8, 7], [8, 6], [10, 7], [10, 6],
    [11, 7], [19, 7], [19, 6], [20, 7], [20, 6]
]

Accuracy = 0.7557377049180328
