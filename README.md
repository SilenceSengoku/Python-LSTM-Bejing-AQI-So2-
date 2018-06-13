# Python-LSTM-Bejing-AQI-So2-
python  and mashine learning

这里是 一组 我的代码，是根据国外大牛对航空客流的预测分析，借鉴而来。

power by 
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

代码 比较简明详细，我还会上传里边的So2数据csv表：“loadDataSo2_4.csv” 这个数据是18年1月1日0时到4月28日23时So2检测数据。
数据来源 ：北京市数据来自北京市环境保护检测中心网站，power by  http://beijingair.sinaapp.com/
根据它发布的数据，我用kettle 抽取清理出了So2这部分的数据进行测试
使用数据的时候需要csv表和代码放在同一文件夹下也就是同一路径下。
当然你也可以选择使用其他数据来试一试，但是你需要记得对数据做预处理（^_^~）

代码环境python 3.6 以及大量的机器学习库，你可以选择下载Anaconda 对应版本 一劳永逸 就是可能会比较占内存，因为我的硬盘 也快极限了 所以我就手动下了 
我需要的的几个库。调出“command” 也就是“cmd”  输入如下命令，如果你是 2 3兼容版本可以在python -py3 来切换pip的2和3版本
python -m pip install pandas
python -m pip install scipy
python -m pip install sklearn
python -m pip install numpy
python -m pip install matplotlib
python -m pip install tensorflow
python -m pip install keras

希望顺利，如果因为2 3版本切换还是卡在更新库这里，可以直接把环境变量里的Path python2暂时删除。再使用pip下载
总之条条大路通罗马。学习的朋友更建议你从power by 那里的网页读一读原文，里边几乎是一步一步的渐进讲解，对每一步理解很有帮助。

这里假设你环境配好，csv表也放在同一目录，双击脚本也好，cmd运行路径也好。

这里说几个简单的修改参数，其他的参数如果想了解，
lr是学习速率，不过代码里没有设定，大概默认为0.001。想设置lr 可以查看优化器optimizers那部分的详细资料
epochs 是迭代,位于model.fit()内的参数
还有一些 保存输出模型的路径需要修改。


最后建议查看keras官网我这里给一个 keras中文官方文档网站：

http://keras-cn.readthedocs.io/en/latest/


通俗来讲 ，我个人认为数值预测问题大部分都是线性回归问题。所以先发聊表纪念今天的进步吧。

