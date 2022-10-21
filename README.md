# WGAN的修改版本
新版本有些函数做了更改,变得不可用，进行了纠正

整个代码做了一些详细的注释，让大家更加方便的去熟悉它

```
python main.py --dataset lsun --dataroot LSUNDIR --cuda
python main.py --mlp_G --ngf 512
generate.py -c samples\generator_config.json -w samples\netG_epoch_3.pth -o imgstest -n 10 --cuda
```
有什么疑问的可以CSDN留言：[WGAN(Wasserstein生成对抗网络)源码的讲解](https://blog.csdn.net/weixin_41896770/article/details/127257347)
