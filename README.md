# LEARN_SVO

## 写在前面
个人认为SVO这个框架还是很好的，然后里面的很多理论知识也是值得去实践的，因此就想重头进行一次彻底的尝试复写；

## 主要的改动
1. 删除了rpg_vikit包的依赖;
2. bundle_adjustment的框架改为了ceres;
3. 2D的alignment,收敛参数改为了0.1,因为0.1个像素确实已经很小了,没有必要那么小;
4. 