# 树
## 面试题46. 把数字翻译成字符串
给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
示例 1:

输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"

题解
用动态规划的思想来解：
先归纳：
12258，从后往前开始，以8来作为基础来翻译，可以翻译成8，58（当然58是不满足的，这里假设满足），那么接下来，只需要翻译1225和122：就把刚才的思路再来一次。
这里需要判断下两位数是否在[10, 25]的范围内。

代码如下
```
class Solution {
public:
	int count = 0;
	int translateNum(int num) {
		if(num < 0)
			return 0;
		calculate(num);
		return count;
	}

	void calculate(int num)
	{
		// 如果num为个位数时
		if (num < 10) {
			count++;
			return;
		}

		// 取出倒数第二位和倒数第一位
		int single = num % 10;
		int two = num % 100;
		calculate(num / 10);
		if(two >= 10 && two <= 25)
		{	
			calculate(num / 100);
		} 
        
	}
};
```  

## 面试题 04.03. 特定深度节点链表  
给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表（比如，若一棵树的深度为 D，则会创建出 D 个链表）。返回一个包含所有深度的链表的数组。  
示例： 

![](C:\picture_tmp\leetcode.PNG)  
题解：  
利用剑指offer里面的解法，即利用一个队列来解，队列q：[root]  
第一次循环时，遍历1次
&emsp; 遍历到root，然后将root的左右子结点放进去，此时q：[2, 3]  
第二次循环时，遍历两次  
&emsp; 遍历到2，将2的子结点放入队列后面，此时q：[3,4,5]   
&emsp; 遍历到3，将3的子结点放入队列后面，此时q：[4,5,7]  
接着第三次，第四次循环  
代码如下:  
```  

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<ListNode*> listOfDepth(TreeNode* tree) {
        vector<ListNode*> result;
        if (tree == NULL)
            return result;
        // 利用队列先进先出的特性
        queue<TreeNode*> q;
        q.push(tree);
        while(!q.empty())
        {
            ListNode* tmp = NULL;
            int size = q.size();
            for(int i = 0; i < size; ++i)
            {
                struct TreeNode* tree_tmp= q.front();
                q.pop();
                ListNode* node = new ListNode(tree_tmp->val);
                if(i == 0)
                // 保存链表头
                {
                    result.push_back(node);
                    tmp = node;
                } else {
                    // 连接链表
                    tmp->next = node;
                    // 保存最后一个结点
                    tmp = node;
                }

                // 将子节点保存
                if(tree_tmp->left) q.push(tree_tmp->left);
                if(tree_tmp->right) q.push(tree_tmp->right);                   
            }
        
        }

        return result;
    }
};
```  
## 739. 每日温度      
### 题目：    
请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。  
### 题解：  
用单调栈来解，  
当 i==0 时，单调栈为空，因此将 0 进栈。

stack=[0(73)]

ans=[0,0,0,0,0,0,0,0]

当 i==1 时，由于 74 大于 73，因此移除栈顶元素 0，赋值 ans[0]:=1-0，将 1 进栈。

stack=[1(74)]

ans=[1,0,0,0,0,0,0,0]

当 i==2 时，由于 75 大于 74，因此移除栈顶元素 1，赋值 ans[1]:=2-1，将 2 进栈。

stack=[2(75)]

ans=[1,1,0,0,0,0,0,0]

当 i==3 时，由于 71 小于 75，因此将 33 进栈。

stack=[2(75),3(71)]

ans=[1,1,0,0,0,0,0,0]

当 i==4 时，由于 69 小于 71，因此将 4 进栈。

stack=[2(75),3(71),4(69)]

ans=[1,1,0,0,0,0,0,0]

当 i==5 时，由于 72 大于 69 和 71，因此依次移除栈顶元素 4 和 3，赋值 ans[4]:=5−4 和 ans[3]:=5−3，将 5 进栈。

stack=[2(75),5(72)]

ans=[1,1,0,2,1,0,0,0]

当 i==6 时，由于 76 大于 72 和 75，因此依次移除栈顶元素 5和 2，赋值 ans[5]:=6−5 和 ans[2]:=6−2，将 6 进栈。

stack=[6(76)]

ans=[1,1,4,2,1,1,0,0]

当 i==7 时，由于 73小于 76，因此将 7 进栈。

stack=[6(76),7(73)]

ans=[1,1,4,2,1,1,0,0]  
### 代码：
```
    vector<int> dailyTemperatures(vector<int>& T) {
        int length = T.size();
        vector<int> result(length);
        stack<int> s;
        for(int i = 0; i < length; ++i)
        {
            while(!s.empty() && T[i] > T[s.top])
            {
                // 如果T[i]大于栈顶，那么栈顶的元素就提出来
                int pos = s.top();
                s.pop();
                result[pos] = i - pos;
            }
            s.push(i);
        }
        
        return result;
    }
```  

## 1302. 层数最深叶子节点的和  
### 题目  
给你一棵二叉树，请你返回层数最深的叶子节点的和。  

### 题解
利用一个辅助队列，先进先出，  
代码如下  
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int deepestLeavesSum(TreeNode* root) {
        // 用一个辅助队列，先进先出
        queue<TreeNode*> queue1;
        int sum = 0;
        if(root == NULL)
            return sum;
        queue1.push(root);
        while(!queue1.empty()){
            sum = 0;
            int length = queue1.size();
            for(int i = 0; i < length; ++i)
            {
                // 先进先出，出队
                TreeNode* tmp = queue1.front();
                sum += tmp->val;
                queue1.pop();
                if(tmp->left)
                    queue1.push(tmp->left);
                if(tmp->right)
                    queue1.push(tmp->right);
            }
        }
        return sum;
    }
};
```

## 894. 所有可能的满二叉树
### 题目  
满二叉树是一类二叉树，其中每个结点恰好有 0 或 2 个子结点。

返回包含 N 个结点的所有可能满二叉树的列表。 答案的每个元素都是一个可能树的根结点。

答案中每个树的每个结点都必须有 node.val=0。

你可以按任何顺序返回树的最终列表。  

### 题解  

对于满二叉树而言，二叉树的总节点数量肯定为奇数。而对于每个节点的左右子树而言，子树的节点数量也肯定为奇数。

满二叉树子树的状态转移规则如下：


f(N) = f(i) + f(N - i - 1)      (i >= 1，i 为奇数，-1 为当前子树的根节点)
例如构建为 5 个节点的满二叉树时，情况有二：左子树为 3 个节点，右子树有 1 个节点；左子树有 1 个节点，右子树有 3 个节点。可以看到实际上是存在重复构建相同数量节点子树的情况。

因此，可以利用后序遍历自下而上构建满二叉树，即先构建节点数量为 1 的子树，后构建节点数量为 3 的子树，再构建节点数量为 5 的子树，直至构建 N 个节点的满二叉树。在每构建一个数量级的节点数量子树时，同时使用 map 记录节点个数与其所有可能的满二叉树形态，减少构建重复的满二叉树。

时间复杂度 O(n)，空间复杂度 O(n)。

### 代码 
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> allPossibleFBT(int N) {
        // 只有奇数才有可能
        vector<TreeNode*> result;
        if((N & 0x1) == 0)
            return result;
        map<int, vector<TreeNode*>> dp;
        result.push_back(new TreeNode(0));
        dp[0] = result;
        // 循环, 即循环3， 5， 7
        for(int i = 3; i <= N; i += 2)
        {
            for(int j = 1; j < i; j += 2)
            {
                vector<TreeNode*> leftNodes = dp[j / 2];
                vector<TreeNode*> rightNodes = dp[(i - j - 1) / 2];
                // 遍历左边的节点
                for(TreeNode* left : leftNodes){
                    // 遍历右边的节点
                    for(TreeNode* right : rightNodes){
                        // 根节点
                        TreeNode* root= new TreeNode(0);
                        root->left = left;
                        root->right = right;
                        dp[i / 2].push_back(root);
                    }
                }
            }
        }
        // dp[0]保存着1个节点的链表根数组
        // dp[1]保存着3个节点的链表根数组
        // dp[2]保存着5个节点的链表根数组
        // dp[3]保存着7个节点的链表根数组
        return dp[N / 2];  
    }
};
```  
## 1305. 两棵二叉搜索树中的所有元素  
### 题目  
给你 root1 和 root2 这两棵二叉搜索树。

请你返回一个列表，其中包含 两棵树 中的所有整数并按 升序 排序。.  
### 代码  
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> getAllElements(TreeNode* root1, TreeNode* root2) {
        // 先转换成数组
        vector<int> vec1, vec2;
        translate(root1, vec1);
        translate(root2, vec2);
        vector<int> result(vec1.size() + vec2.size());
        merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
        return result;
        // 将数组组合起来
        //return combine(vec1, vec2);
    }

    void translate(TreeNode* root, vector<int>& vec)
    {
        if(root == NULL)
            return;
        // 递归左子树
        if(root->left)
            translate(root->left, vec);
        vec.push_back(root->val);
        // 递归右子树
        if(root->right)
            translate(root->right, vec);
    }

};
```  
## 1300. 转变数组后最接近目标值的数组和  
### 题目  
给你一个整数数组 arr 和一个目标值 target ，请你返回一个整数 value ，使得将数组中所有大于 value 的值变成 value 后，数组的和最接近  target （最接近表示两者之差的绝对值最小）。

如果有多种使得和最接近 target 的方案，请你返回这些整数中的最小值。

请注意，答案不一定是 arr 中的数字。

示例 1：

输入：arr = [4,9,3], target = 10
输出：3
解释：当选择 value 为 3 时，数组会变成 [3, 3, 3]，和为 9 ，这是最接近 target 的方案。
示例 2：

输入：arr = [2,3,5], target = 10
输出：5
示例 3：

输入：arr = [60864,25176,27249,21296,20204], target = 56803
输出：11361  
### 题解  
双重二分查找
我们首先考虑题目的一个简化版本：我们需要找到 value，使得数组的和最接近 target 且不大于 target。可以发现，在[0,max(arr)]（即方法一中确定的上下界）的范围之内，随着 value 的增大，数组的和是严格单调递增的。这里「严格」的意思是，不存在两个不同的 value 值，它们对应的数组的和相等。这样一来，一定存在唯一的一个 value 值，使得数组的和接近且大于 target，所以要找到第一个大于target的value。并且由于严格单调递增的性质，我们可以通过二分查找的方法，找到这个 value 值，记为 value_lower。  
### 代码  
```
class Solution {
public:
    int findBestValue(vector<int>& arr, int target) {
        // 先排序
        sort(arr.begin(), arr.end());
        int n = arr.size();
        vector<int> presum(arr.size() + 1); // 初始每个元素为0；
        // 计算出前缀和
        for(int i = 1; i <= n; ++i)
            presum[i] = presum[i - 1] + arr[i - 1];

        int left = 1, right = arr[n - 1]; // 即[1, max]
        // 找第一个大于target的value
        while(left < right)
        {
            int mid = (left + right) / 2;
            int sum = 0;
            
            getsum(arr, presum, mid) < target ? left = mid + 1: right = mid ;
            // 如果sum > target，则最接近target，且小于target的值在前面
        }

        return (abs(getsum(arr, presum, left) - target) < abs(getsum(arr, presum, left - 1) - target)) ? left : left - 1;
    }

    int getsum(vector<int>& arr, vector<int>& presum, int value)
    {
        // 找到第一个大于/等于mid的位置
        vector<int>::iterator iter = lower_bound(arr.begin(), arr.end(), value); // 迭代器
        int sum = presum[iter - arr.begin()] + (arr.end() - iter) * value;
        return sum;
    }
};
```