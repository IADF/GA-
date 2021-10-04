from tkinter import *
import easygui
import time
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
VNS_Num = 1#对前N组进行变领域搜索（对尽可能少的解，进行尽可能仔细的搜索）（）

#快速排序
left=np.zeros((0))
right=np.zeros((0))
left_back=np.zeros((0))

# line_char_arr=[]#
line_char_arr=[]#迭代效果的折线图的y轴数据
line_char_arr_copy1=[]#迭代效果的折线图的y轴数据(单遗传算法)
line_char_arr_copy2=[]#迭代效果的折线图的y轴数据(NEH)
population_Num_arr=[[]]#总群列表
population_Num_arr_copy1=[[]]#总群列表
population_Num_arr_copy2=[[]]#总群列表
population_Num_arr_record=[[]]#禁忌列表
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

#输入（前向排序）
def get_NEH_evaluate(m,t):#计算出评价函数(需要利用迭代的方法来进行计算多工序的时间问题)x：(第几组序列)m：(机器集) t：(时间集)
    global left
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

#后向计算评价指标
def get_back_evaluate(m,t):
    global right
    right=np.zeros((Process_num,len(m)))
    for i in range(Process_num-1,-1,-1):
        for j in range(len(m)-1,-1,-1):
            if i == Process_num-1 and j == len(m)-1:
                right[i][j] = right[i][j]
            elif i == Process_num-1:
                right[i][j] = right[i][j+1] + t[i][m[j+1]-1]
            elif j == len(m)-1:
                right[i][j] = right[i+1][j] + t[i+1][m[j]-1]
            else:
                right[i][j] = max(right[i][j+1] + t[i][m[j+1]-1],right[i+1][j] + t[i+1][m[j]-1])
    return right[0][0]+t[0][m[0]-1]

def bub_evaluate_sort(arr):#排列评价函数
    for i in range(0,len(evaluate_arr)-1):
        for j in range(0,len(evaluate_arr)-1-i):
            if evaluate_arr[j+1]==0:#由于未生成子代因此会产生相关的评价指标为0
                break
            if (evaluate_arr[j]) > (evaluate_arr[j + 1]):
                temp = evaluate_arr[j]
                temp_arr = []
                temp_arr=copy.deepcopy(arr[j])
                arr[j]=arr[j+1]
                arr[j+1]=temp_arr
                evaluate_arr[j] = evaluate_arr[j + 1]
                evaluate_arr[j + 1] = temp
    return 0

# def bub_sort():#冒泡排序算法(排序序列中的元素)
#     for i in range(0,Group_Num):
#         for z in range(0,len(population_Num_arr[i])-1):
#             for j in range(0,len(population_Num_arr[i])-1-z):
#                 if (population_Num_arr[i][j]) > (population_Num_arr[i][j+1]):
#                     temp=population_Num_arr[i][j]
#                     population_Num_arr[i][j]=population_Num_arr[i][j+1]
#                     population_Num_arr[i][j+1]=temp
#     return 0

def part_intersect(arr,z,x,y,begin,end):#种群局部交叉(z,未使用的序列,x，y:交叉操作的两个序列，begin:交叉的起点，end：交叉的终点)

    return_arr = []
    return_arr.append(copy.deepcopy(arr[z]))
    return_arr.append(copy.deepcopy(arr[z+1]))
    #局部片段交叉
    for i in range(begin,end):
        temp=(return_arr[0][i])
        return_arr[0][i]=return_arr[1][i]
        return_arr[1][i]=temp

    #重复检测
    temp_num = []
    for v in range(1, Unit_Num_Limit + 1):
        temp_num.append(v)  # 创建一个1-50的列表，用于防止产生的序列中产生相同的元素
    for j in range(0, Unit_Num_Limit):
        temp = return_arr[0][j]
        if temp == temp_num[temp - 1]:
            temp_num[temp - 1] = 0
        else:#存在重复元素
            return_arr[0][j] = 0#将重复元素置为0，后期重新填入数字
    for i in range(return_arr[0].count(0)):
        return_arr[0][return_arr[0].index(0)]=np.max(temp_num)
        temp_num[temp_num.index(np.max(temp_num))] = 0

    temp_num = []
    for v in range(1, Unit_Num_Limit + 1):
        temp_num.append(v)  # 创建一个1-50的列表，用于防止产生的序列中产生相同的元素
    for j in range(0, Unit_Num_Limit):
        temp = return_arr[1][j]
        if temp == temp_num[temp - 1]:
            temp_num[temp - 1] = 0
        else:#存在重复元素
            return_arr[1][j] = 0#将重复元素置为0，后期重新填入数字
    for i in range(return_arr[1].count(0)):
        return_arr[1][return_arr[1].index(0)]=np.max(temp_num).max()
        temp_num[temp_num.index(np.max(temp_num))]=0

    return return_arr

#选取一个评价数组中最大的元素
#利用最大的元素与所有的元素相减
def roulette_selection():#轮盘赌选出交叉
    roulette_arr=np.zeros(len(evaluate_arr))
    for i in range(0,len(evaluate_arr)-1):
        if evaluate_arr[i] == 0:  # 由于未生成子代因此会产生相关的评价指标为0
            break
        roulette_arr[i]=evaluate_arr.max()-evaluate_arr[i]

    if roulette_arr[0]==0:
        random_val=len(evaluate_arr)-1
        # for i in range(0,len(evaluate_arr)):
        #     if evaluate_arr[i]==0:
        #         random_val=i
        #         break
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
            plt.barh(i+1, t[i][m[j]-1],1, left=left[i][j] , color=color[color_num])
            plt.text(left[i][j] + t[i][m[j]-1] / 8, i+1, str(m[j]), color="white", size=5)
            # plt.yticks(np.arange(Process_num), np.arange(0, Process_num))
    return 0

def variation(arr,z,x):#变异操作(z(变异后产生的子列),x(变异的父列)，variation_point(变异点))单点变异
    #进行单点变异操作
    return_arr=copy.deepcopy(arr[x])
    random.seed()

    variation_point = random.randint(0, Unit_Num_Limit-1)
    variation_val = random.randint(1, Unit_Num_Limit)

    # print("while+++variation_val"+str(variation_val))
    # print("while+++population_Num_arr[x]" + str(population_Num_arr[x]))
    # print("while+++population_Num_arr[z]"+str(population_Num_arr[z]))
    while return_arr.index(variation_val) == variation_point:
        variation_point = random.randint(0,Unit_Num_Limit-1)
        variation_val = random.randint(1,Unit_Num_Limit)

    # print("variation_val"+str(variation_val))
    # print("population_Num_arr[x]" + str(population_Num_arr[x]))
    # print("population_Num_arr[z]"+str(population_Num_arr[z]))
    temp_val = return_arr[variation_point]
    return_arr[return_arr.index(variation_val)]=temp_val
    return_arr[variation_point] = variation_val

    return return_arr

#NEH算法创建一组序列（利用随机数创造的NEH）
def NEH_population_init(unit_num_limit):
    NEH_population_arr = []
    #已随机数形式创建的NEH序列
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

#NEH算法创建一组序列（利用工件的所有工序的最大加工时间来创建NEH）
def NEH_population_init2(unit_num_limit):
    NEH_population_arr = []
    #根据每个工件的全工序加工时间的倒序进行插入
    unit_process_max_time = np.zeros(unit_num_limit)#创建一个数组将存储所有单个工件所有工序加工时间总和
    for i in range(unit_num_limit):#计算每个工件的所有工序时间总长
        for j in range(Process_num):
            unit_process_max_time[i] = unit_process_max_time[i] + process_time_arr[j][i]

    NEH_population_arr.append(list(unit_process_max_time).index(unit_process_max_time.max())+1)
    unit_process_max_time[list(unit_process_max_time).index(unit_process_max_time.max())] = 0

    while(unit_process_max_time.max()!=0):
        NEH_evaluate = np.zeros(len(NEH_population_arr))
        for i in range(len(NEH_population_arr)):
            temp_NEH_population_arr = copy.deepcopy(NEH_population_arr)
            temp_NEH_population_arr.insert(i,list(unit_process_max_time).index(unit_process_max_time.max())+1)
            NEH_evaluate[i]=get_NEH_evaluate(temp_NEH_population_arr,process_time_arr)
        NEH_population_arr.insert(NEH_evaluate.argmin(),list(unit_process_max_time).index(unit_process_max_time.max())+1)
        unit_process_max_time[list(unit_process_max_time).index(unit_process_max_time.max())] = 0
    print("NEH_evaluate.min()",str(NEH_evaluate.min()))
    print("NEH_population_arr" + str(NEH_population_arr))

    return NEH_population_arr

#交换序列中两个元素的位置
def swap_function(arr,x,y):
    temp_return_arr = copy.deepcopy(arr)
    temp_val = temp_return_arr[x]
    temp_return_arr[x] = temp_return_arr[y]
    temp_return_arr[y] = temp_val
    return temp_return_arr

#快速评价
def quick_evaluate(m,t,k,y,insert_val):#arr(机器集)，t（不工件的加工时间），k（进行lift和right计算？），y（插入的位置），insert_val（插入的工件号）
    global left,left_back
    if k==0:
        print("back_evaluate" + str(get_back_evaluate(m,t)))
        print("forward" + str(get_NEH_evaluate(m,t)))
        left_back.resize(len(left),refcheck=False)
        left_back = left
    else:#right(left)[工序][工件]
        left = left_back
        c_arr = np.zeros(Process_num)
        new_insert_arr = np.zeros(Process_num)#计算新插入序列和插入序列后一行序列的时间
        for i in range(Process_num):
            if i == 0:
                new_insert_arr[i] = left[i][y-1] + t[i][m[y-1]-1]
            else:
                new_insert_arr[i] = max(left[i][y-1] + t[i][m[y-1]-1] , new_insert_arr[i-1] + t[i-1][m[y]-1])

        if y == len(m)-1:#将取出的工件插入到了最后
            # print("c_arr.max()" + str(new_insert_arr[i] + t[i][m[y]-1]))
            return new_insert_arr[i] + t[i][m[y]-1]

        for i in range(Process_num):
            c_arr[i] = new_insert_arr[i] + t[i][m[y]-1] + right[i][y] + t[i][m[y+1]-1]

        # print("c_arr.max()" + str(c_arr.max()))
        return c_arr.max()
    return 0

#变领域搜索(仅使用了交换)
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
            print("VNS解同值" + str(get_NEH_evaluate(compare_arr, process_time_arr)))
            k = k - 1

    return part_prepreerence_arr

#变领域搜索（使用插入，使得整个邻域能够尽可能全部遍历）
def VNS_function2(VNS_arr):#VNS_arr（变领域搜索的数组），k（遍历次数）（在一定次数中如果没有最优值没发生变化的话则认为得到的局部最优解）
    k = len(VNS_arr)
    part_prepreerence_arr = copy.deepcopy(VNS_arr)
    part_prepreerence_val = get_NEH_evaluate(part_prepreerence_arr, process_time_arr)# 存储局部最优解的评价函数
    print("VNS前评价值" + str(part_prepreerence_val))

    i = 0
    while i < (k-3):
        compare_arr = copy.deepcopy(part_prepreerence_arr)
        insert_val = compare_arr.pop(i)#弹出一个元素进行插入操作
        # quick_evaluate(compare_arr,process_time_arr,0,0,0)
        for j in range(i+1,len(part_prepreerence_arr)):

            compare_arr.insert(j,insert_val)
            # VNS_evaluate_val=quick_evaluate(compare_arr,process_time_arr,1,j,insert_val)
            VNS_evaluate_val=get_NEH_evaluate(compare_arr,process_time_arr)
            print(VNS_evaluate_val)
            if part_prepreerence_val < VNS_evaluate_val:
                compare_arr.pop(compare_arr.index(insert_val))
                print("VNS评价值" + str(part_prepreerence_val))
            elif part_prepreerence_val > VNS_evaluate_val:
                part_prepreerence_val = VNS_evaluate_val
                part_prepreerence_arr = copy.deepcopy(compare_arr)
                i=0
                print("VNS更新评价值" + str(part_prepreerence_val))
                break
            else:  # 与同值()
                compare_arr.pop(compare_arr.index(insert_val))
                # VNS_function(compare_arr)#将同值序列取出，对两个序列都进行VNS
                print("VNS解同值" + str(get_NEH_evaluate(compare_arr, process_time_arr)))
        i = i + 1

    return part_prepreerence_arr

def changing_population_arr():#加入VNS系统更加容易陷入局部最优解，因此当系统陷入局部最优解的时候进行处理
    #当系统中的评价指标都相同的时候，可以判断解陷入了局部最优解中
    #因此需要增大扰动使得系统跳出最优解，哪怕是一个坏解也没有关系
    #但是需要为解序列添加更多的解可能性，否则系统中的解会进行大量
    #的重复运算导致系统的算力浪费
    return 0


#主函数
if __name__ == '__main__':
    chose_list=[]
    chose_val=0
    while 1:#主循环

        chose_val=easygui.buttonbox("","GA",["初始化种群","NEH+变领域下降迭代","NEH迭代","单遗传算法","生成折线图","生成甘特图","test"],)

        if chose_val=="初始化种群":
            chose_list = easygui.multenterbox("请输入相关参数","初始化种群",("工件数量","工序数","种群数量","创建NEH种群数"),())
            Unit_Num_Limit = int(chose_list[0])
            Process_num = int(chose_list[1])
            Group_Num = int(chose_list[2])
            NEH_NUM = int(chose_list[3])
            process_time_arr.resize((Process_num, Unit_Num_Limit),refcheck=False)  # 不同工序加工时间数组
            population_init(Unit_Num_Limit, Group_Num)  # 第一代种群创建
            evaluate_init()

            #复制种群（不带NEH）
            population_Num_arr_copy1 = copy.deepcopy(population_Num_arr)  # 创建复制种群1（单遗传算法）

            for i in range(NEH_NUM):#利用NEH创建的一组解（屏蔽掉则改为利用随机创建的解）
                population_Num_arr[i]=NEH_population_init(Unit_Num_Limit)#创建一组NEH的解
            population_Num_arr[0]=NEH_population_init2(Unit_Num_Limit)#利用NEH创建一组解（利用新的每个工件所有工序加工时间总和的倒序进行设计）

            #复制种群
            population_Num_arr_copy2 = copy.deepcopy(population_Num_arr)#创建复制种群2（NEH）

            easygui.textbox("生成","显示",str(population_Num_arr)+str(list(process_time_arr)),1)

        elif chose_val=="NEH+变领域下降迭代":
            chose_list = easygui.multenterbox("请输入相关参数", "迭代设置", ("迭代次数","交叉概率","变异概率"), ())
            iteration_num = int(chose_list[0])
            intersect_probability = float(chose_list[1])
            variation_probability = float(chose_list[2])
            k = iteration_num
            line_char_arr.clear()
            evaluate_arr=np.zeros((len(population_Num_arr_copy1) - 1))

            for i in range(0, len(population_Num_arr) - 1):
                evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                get_evaluate(i, population_Num_arr[i], process_time_arr)
                bub_evaluate_sort(population_Num_arr)
            while len(population_Num_arr) - 1 > Group_Num:
                population_Num_arr.pop(Group_Num)
                evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])
            population_Num_arr.pop(population_Num_arr.index([]))

            #NEH+变领域下降
            while k:  # 迭代过程中
                z = len(population_Num_arr) - 1
                while z < Group_Num * 2:
                    temp = random.random()
                    if temp < intersect_probability:
                        x = random.randint(0, Unit_Num_Limit - 1)  # 随机产生两个交叉的点
                        y = random.randint(0, Unit_Num_Limit - 1)
                        part_intersect_arr = part_intersect(population_Num_arr, len(population_Num_arr) - 3, roulette_selection(), roulette_selection(), min(x, y), max(x, y))  # 交叉：：：roulette_selection()轮盘赌函数
                        population_Num_arr.append(copy.deepcopy(part_intersect_arr[0]))
                        population_Num_arr.append(copy.deepcopy(part_intersect_arr[1]))
                        evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                        z = len(population_Num_arr) - 1

                    elif (temp >= intersect_probability) and (temp < intersect_probability + variation_probability):
                        # population_Num_arr.append([])
                        evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                        # print("len(population_Num_arr) - 2:"+str(len(population_Num_arr) - 2))
                        # print("roulette_selection()"+str(a))
                        population_Num_arr.append(variation(population_Num_arr,len(population_Num_arr) - 2,roulette_selection()))
                        z = len(population_Num_arr) - 1

                for i in range(0, len(population_Num_arr) - 1):
                    evaluate_arr.resize(len(population_Num_arr) - 1, refcheck=False)  # 评价指标数组
                    get_evaluate(i, population_Num_arr[i], process_time_arr)
                    bub_evaluate_sort(population_Num_arr)
                while len(population_Num_arr) - 1 > Group_Num:
                    population_Num_arr.pop(Group_Num)
                    evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])

                #对少而精的序列进行变领域搜索
                # population_Num_arr[0] = copy.deepcopy(VNS_function(population_Num_arr[i]))

                start_time = time.time()
                for i in range(len(population_Num_arr)):
                    if population_Num_arr_record.count(population_Num_arr[i]) == 0:
                        population_Num_arr[i] = copy.deepcopy(VNS_function2(population_Num_arr[i]))
                        population_Num_arr_record.append(population_Num_arr[i])#添加禁忌列表(防止重复搜索)
                        break
                end_time = time.time()
                print("消耗时间" + str(end_time - start_time))

                line_char_arr.append(evaluate_arr[0])
                k = k - 1

        elif chose_val=="NEH迭代":
            chose_list = easygui.multenterbox("请输入相关参数", "迭代设置", ("迭代次数","交叉概率","变异概率"), ())
            iteration_num = int(chose_list[0])
            intersect_probability = float(chose_list[1])
            variation_probability = float(chose_list[2])
            k = iteration_num
            line_char_arr_copy2.clear()
            evaluate_arr=np.zeros((len(population_Num_arr_copy1) - 1))

            for i in range(0, len(population_Num_arr_copy2) - 1):
                evaluate_arr.resize(len(population_Num_arr_copy2) - 1, refcheck=False)  # 评价指标数组
                get_evaluate(i, population_Num_arr_copy2[i], process_time_arr)
                bub_evaluate_sort(population_Num_arr_copy2)
            while len(population_Num_arr_copy2) - 1 > Group_Num:
                population_Num_arr_copy2.pop(Group_Num)
                evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])
            population_Num_arr_copy2.pop(population_Num_arr_copy2.index([]))

            while k:  # 迭代过程中
                z = len(population_Num_arr_copy2) - 1
                while z < Group_Num * 2:
                    temp = random.random()
                    if temp < intersect_probability:
                        x = random.randint(0, Unit_Num_Limit - 1)  # 随机产生两个交叉的点
                        y = random.randint(0, Unit_Num_Limit - 1)
                        part_intersect_arr = part_intersect(population_Num_arr_copy2, len(population_Num_arr_copy2) - 3, roulette_selection(), roulette_selection(), min(x, y), max(x, y))  # 交叉：：：roulette_selection()轮盘赌函数
                        population_Num_arr_copy2.append(copy.deepcopy(part_intersect_arr[0]))
                        population_Num_arr_copy2.append(copy.deepcopy(part_intersect_arr[1]))
                        evaluate_arr.resize(len(population_Num_arr_copy2) - 1, refcheck=False)  # 评价指标数组
                        z = len(population_Num_arr_copy2) - 1

                    elif (temp >= intersect_probability) and (temp < intersect_probability + variation_probability):
                        evaluate_arr.resize(len(population_Num_arr_copy2) - 1, refcheck=False)  # 评价指标数组
                        population_Num_arr_copy2.append(variation(population_Num_arr_copy2,len(population_Num_arr_copy2) - 2,roulette_selection()))
                        z = len(population_Num_arr_copy2) - 1

                for i in range(0, len(population_Num_arr_copy2) - 1):
                    evaluate_arr.resize(len(population_Num_arr_copy2) - 1, refcheck=False)  # 评价指标数组
                    get_evaluate(i, population_Num_arr_copy2[i], process_time_arr)
                    bub_evaluate_sort(population_Num_arr_copy2)
                while len(population_Num_arr_copy2) - 1 > Group_Num:
                    population_Num_arr_copy2.pop(Group_Num)
                    evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])

                line_char_arr_copy2.append(evaluate_arr[0])
                k = k - 1


        elif chose_val=="单遗传算法":
            chose_list = easygui.multenterbox("请输入相关参数", "迭代设置", ("迭代次数","交叉概率","变异概率"), ())
            iteration_num = int(chose_list[0])
            intersect_probability = float(chose_list[1])
            variation_probability = float(chose_list[2])
            k = iteration_num
            line_char_arr_copy1.clear()
            evaluate_arr=np.zeros((len(population_Num_arr_copy1) - 1))

            for i in range(0, len(population_Num_arr_copy1) - 1):
                evaluate_arr.resize(len(population_Num_arr_copy1) - 1, refcheck=False)  # 评价指标数组
                get_evaluate(i, population_Num_arr_copy1[i], process_time_arr)
                bub_evaluate_sort(population_Num_arr_copy1)
            while len(population_Num_arr_copy1) - 1 > Group_Num:
                population_Num_arr_copy1.pop(Group_Num)
                evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])
            population_Num_arr_copy1.pop(population_Num_arr_copy1.index([]))

            while k:  # 迭代过程中
                z = len(population_Num_arr_copy1) - 1
                while z < Group_Num * 2:
                    temp = random.random()
                    if temp < intersect_probability:
                        x = random.randint(0, Unit_Num_Limit - 1)  # 随机产生两个交叉的点
                        y = random.randint(0, Unit_Num_Limit - 1)
                        part_intersect_arr = part_intersect(population_Num_arr_copy1, len(population_Num_arr_copy1) - 3, roulette_selection(), roulette_selection(), min(x, y), max(x, y))  # 交叉：：：roulette_selection()轮盘赌函数
                        population_Num_arr_copy1.append(part_intersect_arr[0])
                        population_Num_arr_copy1.append(part_intersect_arr[1])
                        evaluate_arr.resize(len(population_Num_arr_copy1) - 1, refcheck=False)  # 评价指标数组
                        z = len(population_Num_arr_copy1) - 1

                    elif (temp >= intersect_probability) and (temp < intersect_probability + variation_probability):
                        evaluate_arr.resize(len(population_Num_arr_copy1) - 1, refcheck=False)  # 评价指标数组
                        population_Num_arr_copy1.append(variation(population_Num_arr_copy1,len(population_Num_arr_copy1) - 2,roulette_selection()))
                        z = len(population_Num_arr_copy1) - 1

                for i in range(0, len(population_Num_arr_copy1) - 1):
                    evaluate_arr.resize(len(population_Num_arr_copy1) - 1, refcheck=False)  # 评价指标数组
                    get_evaluate(i, population_Num_arr_copy1[i], process_time_arr)
                    bub_evaluate_sort(population_Num_arr_copy1)
                while len(population_Num_arr_copy1) - 1 > Group_Num:
                    population_Num_arr_copy1.pop(Group_Num)
                    evaluate_arr = np.delete(evaluate_arr, [Group_Num, ])

                line_char_arr_copy1.append(evaluate_arr[0])
                k = k - 1

        elif chose_val=="生成折线图":
            x = range(len(line_char_arr))
            #plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度，线的宽度和标签
            plt.plot(x,line_char_arr,'ro-',color='blue',alpha=0.8,linewidth=1,label="emmmmm")
            x = range(len(line_char_arr_copy2))
            plt.plot(x, line_char_arr_copy2, 'ro-', color='red', alpha=0.8, linewidth=1, label="emmmmm")
            x = range(len(line_char_arr_copy1))
            plt.plot(x, line_char_arr_copy1, 'ro-', color='green', alpha=0.8, linewidth=1, label="emmmmm")
            plt.show()

        elif chose_val=="生成甘特图":#只生成最后一组用时最短的序列
            gatt(population_Num_arr[0], process_time_arr)

            # plt.yticks(np.arange(max(m)), np.arange(1, max(m) + 1))
            plt.show()
            print(evaluate_arr[0],evaluate_arr[1])

        elif chose_val=="test":
            NEH_population_init2(Unit_Num_Limit)
