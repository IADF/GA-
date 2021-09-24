from tkinter import *
import easygui
import psutil
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sys
import math

sys.setrecursionlimit(2000)

#参数设置
Unit_Num_Limit = 20#创建序列的长度
Group_Num = 20#创建多少组序列
Process_time_up_limit = 99#工序加工时间上界
Process_time_down_limit = 50#工序加工时间下界
Process_num = 20#工序数
VNS_Num = 20#对前N组进行变领域搜索

line_char_arr=[]#
line_char_arr=[]#迭代效果的折线图的y轴数据
population_Num_arr=[[]]#总群列表
process_time_arr=np.zeros((Process_num,Unit_Num_Limit))#不同工序加工时间数组
evaluate_arr=np.zeros(Group_Num*2)#评价指标数组
unit_complete_arr=np.zeros((Unit_Num_Limit,Process_num))#每个工件在不同工序的完工时间

def population_init(unit_num_limit,group_num):#种群初始化函数（不同工序的排列生成初始解）
    for i in range(0,group_num):
        population_Num_arr.append([])
        temp_num=[]
        for v in range(1, Unit_Num_Limit+1):
            temp_num.append(v)  # 创建一个1-50的列表，用于防止产生的序列中产生相同的元素
        for j in range(0,unit_num_limit):
            while True:#重复元素判断
                random.seed()
                temp = random.randint(1, unit_num_limit)  # 生成1-50的随机数
                if temp==temp_num[temp-1]:
                    population_Num_arr[i].append(temp)
                    temp_num[temp-1]=0
                    break
    # for i in range(0,group_num):
    #     population_Num_arr.append([])
    # print(population_Num_arr)
    return population_Num_arr

def evaluate_init():#初始化函数(每个工件在不同工序对应的时间)
    for i in range(0,Process_num):#工序数
        for j in range(0,Unit_Num_Limit):
            random.seed()
            temp = random.randint(Process_time_down_limit, Process_time_up_limit)  # 生成1-50的随机数
            process_time_arr[i][j]=temp
    # for i in range(100,200):
    #     evaluate_arr[i]=999999
    # print(process_time_arr)
    return 0

#输入
def get_evaluate(x,m,t):#计算出评价函数(需要利用迭代的方法来进行计算多工序的时间问题)x：(第几组序列)m：(机器集) t：(时间集)
    left=np.zeros((Process_num,Unit_Num_Limit))
    for i in range(Process_num):
        for j in range(Unit_Num_Limit):
            if i==0 and j==0:
                left[i][j] = 0
            elif i == 0 :
                left[i][j] = left[i][j-1] + t[i][m[j-1]-1]
            elif j == 0:
                left[i][j] = left[i-1][0] + t[i-1][m[j]-1]
            else :
                left[i][j] = max(left[i][j-1]+t[i][m[j-1]-1],left[i-1][j]+t[i-1][m[j]-1])
    evaluate_arr[x]=left[Process_num-1][Unit_Num_Limit-1]+t[Process_num-1][m[Unit_Num_Limit-1]-1]
    return 0

#输入
def get_NEH_evaluate(m,t):#计算出评价函数(需要利用迭代的方法来进行计算多工序的时间问题)x：(第几组序列)m：(机器集) t：(时间集)
    left=np.zeros((Process_num,len(m)))
    for i in range(Process_num):
        for j in range(len(m)):
            if i==0 and j==0:
                left[i][j] = 0
            elif i == 0 :
                left[i][j] = left[i][j-1] + t[i][m[j-1]-1]
            elif j == 0:
                left[i][j] = left[i-1][0] + t[i-1][m[j]-1]
            else :
                left[i][j] = max(left[i][j-1]+t[i][m[j-1]-1],left[i-1][j]+t[i-1][m[j]-1])
    return left[Process_num-1][len(m)-1]+t[Process_num-1][m[len(m)-1]-1]

def bub_evaluate_sort():#排列评价函数
    for i in range(0,len(evaluate_arr)-1):
        for j in range(0,len(evaluate_arr)-1-i):
            if evaluate_arr[j+1]==0:#由于未生成子代因此会产生相关的评价指标为0
                break
            if (evaluate_arr[j]) > (evaluate_arr[j + 1]):
                temp = evaluate_arr[j]
                temp_arr = []
                temp_arr=copy.deepcopy(population_Num_arr[j])
                population_Num_arr[j]=population_Num_arr[j+1]
                population_Num_arr[j+1]=temp_arr
                evaluate_arr[j] = evaluate_arr[j + 1]
                evaluate_arr[j + 1] = temp
    return 0

def bub_sort():#冒泡排序算法(排序序列中的元素)
    for i in range(0,Group_Num):
        for z in range(0,len(population_Num_arr[i])-1):
            for j in range(0,len(population_Num_arr[i])-1-z):
                if (population_Num_arr[i][j]) > (population_Num_arr[i][j+1]):
                    temp=population_Num_arr[i][j]
                    population_Num_arr[i][j]=population_Num_arr[i][j+1]
                    population_Num_arr[i][j+1]=temp
    return 0

def part_intersect(z,x,y,begin,end):#种群局部交叉(z,未使用的序列,x，y:交叉操作的两个序列，begin:交叉的起点，end：交叉的终点)

    population_Num_arr[z] = copy.deepcopy(population_Num_arr[x])
    population_Num_arr[z+1] = copy.deepcopy(population_Num_arr[y])
    #局部片段交叉
    for i in range(begin,end):
        temp=(population_Num_arr[z][i])
        population_Num_arr[z][i]=population_Num_arr[z+1][i]
        population_Num_arr[z+1][i]=temp

    #重复检测
    temp_num = []
    for v in range(1, Unit_Num_Limit + 1):
        temp_num.append(v)  # 创建一个1-50的列表，用于防止产生的序列中产生相同的元素
    for j in range(0, Unit_Num_Limit):
        temp = population_Num_arr[z][j]
        if temp == temp_num[temp - 1]:
            temp_num[temp - 1] = 0
        else:#存在重复元素
            population_Num_arr[z][j] = 0#将重复元素置为0，后期重新填入数字
    for i in range(population_Num_arr[z].count(0)):
        population_Num_arr[z][population_Num_arr[z].index(0)]=np.max(temp_num)
        temp_num[temp_num.index(np.max(temp_num))] = 0


    temp_num = []
    for v in range(1, Unit_Num_Limit + 1):
        temp_num.append(v)  # 创建一个1-50的列表，用于防止产生的序列中产生相同的元素
    for j in range(0, Unit_Num_Limit):
        temp = population_Num_arr[z + 1][j]
        if temp == temp_num[temp - 1]:
            temp_num[temp - 1] = 0
        else:#存在重复元素
            population_Num_arr[z + 1][j] = 0#将重复元素置为0，后期重新填入数字
    for i in range(population_Num_arr[z+1].count(0)):
        population_Num_arr[z + 1][population_Num_arr[z+1].index(0)]=np.max(temp_num).max()
        temp_num[temp_num.index(np.max(temp_num))]=0


    return 0

#选取一个评价数组中最大的元素
#利用最大的元素与所有的元素相减
def roulette_selection():#轮盘赌选出交叉
    roulette_arr=np.zeros(len(evaluate_arr))
    for i in range(0,len(evaluate_arr)-1):
        if evaluate_arr[i] == 0:  # 由于未生成子代因此会产生相关的评价指标为0
            break
        roulette_arr[i]=evaluate_arr.max()-evaluate_arr[i]

    if roulette_arr[0]==0:
        random_val=0
        for i in range(0,len(evaluate_arr)):
            if evaluate_arr[i]==0:
                random_val=i
                break
        random.seed()
        temp = random.randint(0,random_val-1 )  # 创建一个随机变量用于轮盘赌的概率选择
        print("*************************")
        return temp
    random.seed()
    temp=random.randint(0,roulette_arr.sum())#创建一个随机变量用于轮盘赌的概率选择
    for i in range(0,len(evaluate_arr)-1):
        temp = temp - roulette_arr[i]
        print("temp:"+str(temp))
        if temp - roulette_arr[i] < 0:
            print("i:" + str(i))
            return i
        print("roulette_arr:"+str(roulette_arr))
        print("roulette_arr.sum():"+str(roulette_arr.sum()))
    return -1#如果返回-1则程序有问题

def gatt(m,t):#甘特图   m：(机器集) t：(时间集)left
    left=np.zeros((Process_num,Unit_Num_Limit))
    for i in range(Process_num):
        for j in range(Unit_Num_Limit):
            if i==0 and j==0:
                left[i][j] = 0
            elif i == 0 :
                left[i][j] = left[i][j-1] + t[i][m[j-1]-1]
            elif j == 0:
                left[i][j] = left[i-1][0] + t[i-1][m[j]-1]
            else :
                left[i][j] = max(left[i][j-1]+t[i][m[j-1]-1],left[i-1][j]+t[i-1][m[j]-1])

    color = ['b', 'g', 'r', 'y', 'c', 'm', 'k']
    for i in range(0,Process_num):
        for j in range(0,Unit_Num_Limit):
            color_num = j % 7
            plt.barh(i+1, t[i][m[j]-1], left=left[i][j] , color=color[color_num])
    return 0

def variation(z,x):#变异操作(z(变异后产生的子列),x(变异的父列)，variation_point(变异点))单点变异
    #进行单点变异操作
    population_Num_arr[z]=population_Num_arr[x]
    random.seed()

    variation_point = random.randint(0, Unit_Num_Limit-1)
    variation_val = random.randint(1, Unit_Num_Limit)

    # print("while+++variation_val"+str(variation_val))
    # print("while+++population_Num_arr[x]" + str(population_Num_arr[x]))
    # print("while+++population_Num_arr[z]"+str(population_Num_arr[z]))
    while population_Num_arr[z].index(variation_val) == variation_point:
        variation_point = random.randint(0,Unit_Num_Limit-1)
        variation_val = random.randint(1,Unit_Num_Limit)

    # print("variation_val"+str(variation_val))
    # print("population_Num_arr[x]" + str(population_Num_arr[x]))
    # print("population_Num_arr[z]"+str(population_Num_arr[z]))
    temp_val = population_Num_arr[z][variation_point]
    population_Num_arr[z][population_Num_arr[z].index(variation_val)]=temp_val
    population_Num_arr[z][variation_point] = variation_val

    return 0

#NEH算法创建一组序列
def NEH_population_init(unit_num_limit):
    NEH_population_arr = []
    duplicate_detection = [0]*Unit_Num_Limit
    random_val = random.randint(1, unit_num_limit)
    duplicate_detection[random_val-1] = 1
    NEH_population_arr.append(random_val)

    for z in range(Unit_Num_Limit-1):
        NEH_evaluate = np.zeros(len(NEH_population_arr))
        while True:  # 重复元素判断
            random_val = random.randint(1, unit_num_limit)  # 生成1-50的随机数
            if duplicate_detection[random_val-1]==0:
                duplicate_detection[random_val-1] = 1
                break
        for i in range(len(NEH_population_arr)):
            temp_NEH_population_arr = copy.deepcopy(NEH_population_arr)
            temp_NEH_population_arr.insert(i,random_val)
            NEH_evaluate[i]=get_NEH_evaluate(temp_NEH_population_arr,process_time_arr)
        NEH_population_arr.insert(NEH_evaluate.argmin(),random_val)
    print("NEH_evaluate.min()",str(NEH_evaluate.min()))

    print(NEH_population_arr)
    return NEH_population_arr

#交换序列中两个元素的位置
def swap_function(arr,x,y):
    temp_return_arr = copy.deepcopy(arr)
    temp_val = temp_return_arr[x]
    temp_return_arr[x] = temp_return_arr[y]
    temp_return_arr[y] = temp_val
    return temp_return_arr

#变领域搜索
def VNS_function(VNS_arr):#VNS_arr（变领域搜索的数组），k（遍历次数）（在一定次数中如果没有最优值没发生变化的话则认为得到的局部最优解）
    k = len(VNS_arr)
    part_prepreerence_arr = VNS_arr
    part_prepreerence_val = get_NEH_evaluate(part_prepreerence_arr,process_time_arr)#存储局部最优解的评价函数
    print("VNS前评价值" + str(part_prepreerence_val))
    while k:
        compare_arr = swap_function(part_prepreerence_arr, k - 1, k - 2)
        VNS_evaluate_val = get_NEH_evaluate(compare_arr, process_time_arr)
        if part_prepreerence_val < VNS_evaluate_val:
            k = k - 1
        elif part_prepreerence_val > VNS_evaluate_val:
            part_prepreerence_val = VNS_evaluate_val
            part_prepreerence_arr = copy.deepcopy(compare_arr)
            k = len(part_prepreerence_arr)
            print("VNS评价值" + str(part_prepreerence_val))
        else: #与同值()
            # VNS_function(compare_arr)#将同值序列取出，对两个序列都进行VNS
            print("VNS解同值"+str(get_NEH_evaluate(compare_arr, process_time_arr)))
            k = k-1


    return part_prepreerence_arr

def changing_population_arr():#加入VNS系统更加容易陷入局部最优解，因此当系统陷入局部最优解的时候进行处理

    return 0


#主函数
if __name__ == '__main__':
    chose_list=[]
    chose_val=0
    while 1:#主循环

        chose_val=easygui.buttonbox("","GA",["初始化种群","迭代","生成折线图","生成甘特图","test"],)

        if chose_val=="初始化种群":
            chose_list = easygui.multenterbox("请输入相关参数","初始化种群",("工件数量","工序数","种群数量","创建NEH种群数"),())
            Unit_Num_Limit = int(chose_list[0])
            Process_num = int(chose_list[1])
            Group_Num = int(chose_list[2])
            NEH_NUM = int(chose_list[3])
            process_time_arr.resize((Process_num, Unit_Num_Limit),refcheck=False)  # 不同工序加工时间数组
            population_init(Unit_Num_Limit, Group_Num)  # 第一代种群创建
            evaluate_init()

            # for i in range(0, len(population_Num_arr) - 1):
            #     evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
            #     get_evaluate(i, population_Num_arr[i], process_time_arr)
            #     bub_evaluate_sort()
            # while len(population_Num_arr) - 1 > Group_Num:
            #     population_Num_arr.pop(Group_Num)
            #     evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])
            # print("average"+str(evaluate_arr.sum()/len(evaluate_arr)))

            for i in range(NEH_NUM):#利用NEH创建的一组解（屏蔽掉则改为利用随机创建的解）
                population_Num_arr[i]=NEH_population_init(Unit_Num_Limit)#创建一组NEH的解
            for i in range(len(population_Num_arr)-1):
                print("population_Num_arr_evaluate" + str(i) + "=" + str(get_NEH_evaluate(population_Num_arr[i],process_time_arr)))
            # for i in range(0, len(population_Num_arr) - 1):
            #     evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
            #     get_evaluate(i, population_Num_arr[i], process_time_arr)
            #     bub_evaluate_sort()
            # while len(population_Num_arr) - 1 > Group_Num:
            #     population_Num_arr.pop(Group_Num)
            #     evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])
            # print("average_NEH" + str(evaluate_arr.sum() / len(evaluate_arr)))

            easygui.textbox("生成","显示",str(population_Num_arr)+str(list(process_time_arr)),1)

        elif chose_val=="迭代":
            chose_list = easygui.multenterbox("请输入相关参数", "迭代设置", ("迭代次数","交叉概率","变异概率"), ())
            iteration_num = int(chose_list[0])
            intersect_probability = float(chose_list[1])
            variation_probability = float(chose_list[2])
            k = iteration_num
            line_char_arr.clear()

            for i in range(0, len(population_Num_arr) - 1):
                evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                get_evaluate(i, population_Num_arr[i], process_time_arr)
                bub_evaluate_sort()
            while len(population_Num_arr) - 1 > Group_Num:
                population_Num_arr.pop(Group_Num)
                evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])

            while k:  # 迭代过程中
                z = len(population_Num_arr) - 1
                while z < Group_Num * 2:
                    temp = random.random()
                    if temp < intersect_probability:
                        x = random.randint(0, Unit_Num_Limit - 1)  # 随机产生两个交叉的点
                        y = random.randint(0, Unit_Num_Limit - 1)
                        population_Num_arr.append([])
                        population_Num_arr.append([])
                        evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                        part_intersect(len(population_Num_arr) - 3, roulette_selection(), roulette_selection(), min(x, y),max(x, y))  # 交叉：：：roulette_selection()轮盘赌函数
                        z = len(population_Num_arr) - 1

                    elif (temp >= intersect_probability) and (temp < intersect_probability + variation_probability):
                        population_Num_arr.append([])
                        evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                        # print("len(population_Num_arr) - 2:"+str(len(population_Num_arr) - 2))
                        # print("roulette_selection()"+str(a))
                        variation(len(population_Num_arr) - 2,roulette_selection())
                        z = len(population_Num_arr) - 1

                for i in range(0, len(population_Num_arr) - 1):
                    evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                    get_evaluate(i, population_Num_arr[i], process_time_arr)
                    bub_evaluate_sort()
                while len(population_Num_arr) - 1 > Group_Num:
                    population_Num_arr.pop(Group_Num)
                    evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])
                for i in range(0,VNS_Num):
                    population_Num_arr[i] = copy.deepcopy(VNS_function(population_Num_arr[i]))
                line_char_arr.append(evaluate_arr[0])
                k = k - 1

        elif chose_val=="生成折线图":
            x = range(len(line_char_arr))
            #plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度，线的宽度和标签
            plt.plot(x,line_char_arr,'ro-',color='blue',alpha=0.8,linewidth=1,label="emmmmm")
            plt.show()

        elif chose_val=="生成甘特图":#只生成最后一组用时最短的序列
            gatt(population_Num_arr[0], process_time_arr)

            # plt.yticks(np.arange(max(m)), np.arange(1, max(m) + 1))
            plt.show()
            print(evaluate_arr[0],evaluate_arr[1])

        elif chose_val=="test":
            NEH_population_init(Unit_Num_Limit)





