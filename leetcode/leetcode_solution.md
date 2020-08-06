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
## 173. 二叉搜索树迭代器  
### 题目  
实现一个二叉搜索树迭代器。你将使用二叉搜索树的根节点初始化迭代器。

调用 next() 将返回二叉搜索树中的下一个最小的数。  
BSTIterator iterator = new BSTIterator(root);
iterator.next();    // 返回 3
iterator.next();    // 返回 7
iterator.hasNext(); // 返回 true
iterator.next();    // 返回 9
iterator.hasNext(); // 返回 true
iterator.next();    // 返回 15
iterator.hasNext(); // 返回 true
iterator.next();    // 返回 20
iterator.hasNext(); // 返回 false

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
class BSTIterator {
public:
    // 用双链表存起来
    //TreeNode* listHead;
    TreeNode* curNode;
    BSTIterator(TreeNode* root) {
        if(root == NULL)
        {
            //listHead = NULL;
            curNode = NULL;
        } else {
           TreeNode* tmp = NULL;
           TreeNode* cur = root;
           translate(root, &tmp);
           while(cur->left != NULL)
           {
               cur = cur->left;
           }

           //listHead = cur;
           curNode = cur;
        }
    }
    
    /** @return the next smallest number */
    int next() {
        int temp = curNode->val;
        curNode = curNode->right;
        return temp;
    }
    
    /** @return whether we have a next smallest number */
    bool hasNext() {
        return (curNode == NULL) ? false : true;
    }

    void translate(TreeNode* root, TreeNode** lastNode)
    {
        //先把左子树的链表排序
        if(root->left)
            translate(root->left, lastNode);
        // 排完序后
        root->left = *lastNode;
        if(*lastNode)
            (*lastNode)->right = root;
        // 最左边的节点就是当前的root
        (*lastNode) = root;
        if(root->right)
            translate(root->right, lastNode);
    }
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
```  
## 94. 二叉树的中序遍历  
### 题目  
给定一个二叉树，返回它的中序 遍历。  用迭代算法来解  
### 题解   
```
栈S;  
p= root;  
while(p || S不空){  
    while(p){  
        p入S;  
        p = p的左子树;  
    }  
    p = S.top 出栈;  
    访问p;  
    p = p的右子树;  
}
```  
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
    vector<int> inorderTraversal(TreeNode* root) {
        // 利用一个栈,先进后出
        vector<int> result;
        stack<TreeNode*> s;
        TreeNode* tmp = root;
        while(tmp != NULL || !s.empty())
        {
            // 先将左子树压栈
            while(tmp)
            {
                s.push(tmp);
                tmp = tmp->left;
            }
            // 左子树压栈完后，再拿出来遍历
            tmp = s.top();
            s.pop();
            result.push_back(tmp->val);
            // 取出右子树
            tmp = tmp->right;
        }
        return result;
    }
};
```  
## 14. 最长公共前缀  
### 题目  
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。  
示例 1:

输入: ["flower","flow","flight"]  
输出: "fl"  
示例 2:  

输入: ["dog","racecar","car"]  
输出: ""  
解释: 输入不存在公共前缀。  

### 题解  
横向扫描
LCP(S1 … Sn) 表示字符串S1 … Sn的最长公共前缀。可以得到以下结论：LCP(S1 ... Sn) = LCP(LCP(S1, S2),S3), ... Sn)
基于该结论，可以得到一种查找字符串数组中的最长公共前缀的简单方法。依次遍历字符串数组中的每个字符串，对于每个遍历到的字符串，更新最长公共前缀，当遍历完所有的字符串以后，即可得到字符串数组中的最长公共前缀

### 代码  
```
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int length = strs.size();
        if(!length)
            return "";
        string prefix = strs[0];
        for(int i = 1; i < length; ++i)
        {
            // 求第一个和第二个的prefix
            prefix = longestCommonPrefix(prefix, strs[i]);
            if(!prefix.size()){
                break;
            }   
        }
        return prefix;
    }
    string longestCommonPrefix(string& str1, string& str2)
    {
        int length = min(str1.size(), str2.size());
        int index = 0;
        while(index < length && str1[index] == str2[index])
        {
            ++index;
        }
        return str1.substr(0, index);

    }

};
```

## 513. 找树左下角的值  
### 题目  
给定一个二叉树，在树的最后一行找到最左边的值。  
### 题解  
广度优先遍历，只不过先存入右节点，然后再存入左节点，遍历结束，最后一个节点刚好是答案
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
    int findBottomLeftValue(TreeNode* root) {
        if(root == NULL)
            return 0;
        queue<TreeNode*> queue1;
        queue1.push(root);
        TreeNode* tmp = NULL;
        while(!queue1.empty())
        {
            tmp = queue1.front();
            queue1.pop();
            // 先将右边的节点入栈
            if(tmp->right)
                queue1.push(tmp->right);
            if(tmp->left)
                queue1.push(tmp->left);
        }
        return tmp->val;
    }
};
```

## 1325. 删除给定值的叶子节点  
### 题目  
给你一棵以 root 为根的二叉树和一个整数 target ，请你删除所有值为 target 的 叶子节点 。

注意，一旦删除值为 target 的叶子节点，它的父节点就可能变成叶子节点；如果新叶子节点的值恰好也是 target ，那么这个节点也应该被删除。

也就是说，你需要重复此过程直到不能继续删除。
### 代码  
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* removeLeafNodes(TreeNode* root, int target) {
        if(root == NULL)
            return NULL;
        return do_remove(root, target);
    }

    TreeNode* do_remove(TreeNode* root, int target)
    {
        // 如果有左子树, 左子树搞完
        if(root->left)
            root->left = do_remove(root->left, target);
        // 如果有右子树，右子树搞完
        if(root->right)
            root->right = do_remove(root->right, target);

        // 如果左子树和右子树都被干完后, 判断根的值是否为target
        if(root->left == NULL && root->right == NULL)
            if(root->val == target)
                return NULL;
        return root;
    }
    
};
```
## 面试题 04.08. 首个共同祖先
### 题目  
  
设计并实现一个算法，找出二叉树中某两个节点的第一个共同祖先。不得将其他的节点存储在另外的数据结构中。注意：这不一定是二叉搜索树。

例如，给定如下二叉树: root = [3,5,1,6,2,0,8,null,null,7,4]

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
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // 先判断是否为空
        if(root == NULL)
            return NULL;
        // 用递归来做，一直往下递归，找到p/q后就返回，
        // 寻找到的节点必然在根节点上，要么是最初的root，要么就是p/q
        if(root == q || root == p)
            return root;
        // 两种情况: 两个节点在两个左右子树上；两个节点在一个左右子树上；
        // 在两个左右子树上，直接返回root
        // 看左子树有没有
        TreeNode* leftNode = lowestCommonAncestor(root->left, p, q);
        // 看右子树有没有
        TreeNode* rightNode = lowestCommonAncestor(root->right, p, q);
        // 在左右子树上
        if(leftNode != NULL && rightNode != NULL)
            return root;                  // 直接返回当前根节点
        else {
            // 如果有一个为NULL
            if(leftNode != NULL)
                return leftNode;
            else if(rightNode != NULL)
                return rightNode;
            else
                return NULL;
        }

    }
};
```
## 979. 在二叉树中分配硬币
### 题目  
给定一个有 N 个结点的二叉树的根结点 root，树中的每个结点上都对应有 node.val 枚硬币，并且总共有 N 枚硬币。

在一次移动中，我们可以选择两个相邻的结点，然后将一枚硬币从其中一个结点移动到另一个结点。(移动可以是从父结点到子结点，或者从子结点移动到父结点。)。

返回使每个结点上只有一枚硬币所需的移动次数。

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
    int res = 0;
    int distributeCoins(TreeNode* root) {
        if(root == NULL)
            return 0;
        dfs(root);
        return res;
    }

    int dfs(TreeNode* root)
    {
        if(root == NULL)
            return 0;
        int left = dfs(root->left);   // 左子树给根节点多少个硬币; 如果为负数，说明根节点应给左子树
        int right = dfs(root->right); // 右子树给根节点多少个硬币
        res += abs(left) + abs(right);
        return left + right + root->val - 1;
    }
};
```
## 1123. 最深叶节点的最近公共祖先  
### 题目  
给你一个有根节点的二叉树，找到它最深的叶节点的最近公共祖先。

回想一下：

叶节点 是二叉树中没有子节点的节点
树的根节点的 深度 为 0，如果某一节点的深度为 d，那它的子节点的深度就是 d+1
如果我们假定 A 是一组节点 S 的 最近公共祖先，S 中的每个节点都在以 A 为根节点的子树中，且 A 的深度达到此条件下可能的最大值。
 

示例 1：

输入：root = [1,2,3]
输出：[1,2,3]
解释： 
最深的叶子是值为 2 和 3 的节点。
这些叶子的最近共同祖先是值为 1 的节点。
返回的答案为序列化的 TreeNode 对象（不是数组）"[1,2,3]" 。
示例 2：

输入：root = [1,2,3,4]
输出：[4]
示例 3：

输入：root = [1,2,3,4,5]
输出：[2,4,5]

### 题解  
这个题目的例子有点搞混人，其实就是要求最深的叶子节点的最近公共父节点。如果最深的叶子节点没有兄弟，那么公共父节点就是它自己，否则返回它的父节点。
思路：我们可以从某个节点A为根节点的子树思考，先求它左、右节点的高度和最近公共父节点。
1、如果左、右节点的高度相同，那么公共父节点就是A。
2、如果左节点的高度比右节点大，最深的叶子节点的最近公共父节点肯定是左节点返回的结果里面。
3、如果右节点的高度比左节点大，最深的叶子节点的最近公共父节点肯定是右节点返回的结果里面。
4、上面的每一步，都会导致高度加1.
5、如果是空节点，那么高度为0.公共父节点为NULL。

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
    
    typedef pair<int, TreeNode*> Tr;

    Tr leavesDFS(TreeNode* root) {
        if (root == NULL) {
            return Tr(0, NULL);
        }
        auto l = leavesDFS(root->left);
        auto r = leavesDFS(root->right);
        if (l.first == r.first) {
            return Tr(l.first + 1, root);
        } else if (l.first > r.first) {
            return Tr(l.first + 1, l.second);
        } else {
            return Tr(r.first + 1, r.second);
        }
    }

    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        return leavesDFS(root).second;
    }

};
```
## 144. 二叉树的前序遍历
### 题目
给定一个二叉树，返回它的 前序 遍历。不用递归
### 题解
递归的本质就是压栈，了解递归本质后就完全可以按照递归的思路来迭代。
怎么压，压什么？压的当然是待执行的内容，后面的语句先进栈，所以进栈顺序就决定了前中后序。

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
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode*> tmp;
        vector<int> res;
        if(root != NULL)
            tmp.push(root);
        while(!tmp.empty())
        {
            // 先拿出栈顶的数据,并出栈
            TreeNode* root = tmp.top();
            tmp.pop();
            res.push_back(root->val);
            // 由于是先序遍历，所以将右节点先压栈
            if(root->right != NULL)
                tmp.push(root->right);
            // 将左节点后压栈
            if(root->left != NULL)
                tmp.push(root->left);
        }
        return res;
    }
};
```

## 96. 不同的二叉搜索树
### 题目  

给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？

### 题解  
本问题可以用动态规划求解。

给定一个有序序列 1 ... n，为了根据序列构建一棵二叉搜索树。我们可以遍历每个数字 i，将该数字作为树根，1 ... (i-1) 序列将成为左子树，(i+1) ... n 序列将成为右子树。于是，我们可以递归地从子序列构建子树。
在上述方法中，由于根各自不同，每棵二叉树都保证是独特的。

可见，问题可以分解成规模较小的子问题。因此，我们可以存储并复用子问题的解，而不是递归的（也重复的）解决这些子问题，这就是动态规划法。

算法

问题是计算不同二叉搜索树的个数。为此，我们可以定义两个函数：

G(n): 长度为n的序列的不同二叉搜索树个数。

F(i, n): 以i为根的不同二叉搜索树个数(1≤i≤n)。

可见，

G(n) 是我们解决问题需要的函数。

稍后我们将看到，G(n) 可以从 F(i, n) 得到，而 F(i, n) 又会递归的依赖于G(n)。首先，根据上一节中的思路，不同的二叉搜索树的总数 G(n)，是对遍历所有 i (1 <= i <= n) 的 F(i, n) 之和。换而言之：G(n) = ∑F(i,n)  (1 <= i <= n)

特别的，对于边界情况，当序列长度为 1 （只有根）或为 0 （空树）时，只有一种情况。亦即：G(0) = 1, G(1) = 1。给定序列 1 ... n，我们选出数字 i 作为根，则对于根 i 的不同二叉搜索树数量 F(i, n)，是左右子树个数的笛卡尔积。 举例而言，F(3,7)，以 3 为根的不同二叉搜索树个数。为了以 3 为根从序列 [1, 2, 3, 4, 5, 6, 7] 构建二叉搜索树，我们需要从左子序列 [1, 2] 构建左子树，从右子序列 [4, 5, 6, 7] 构建右子树，然后将它们组合(即笛卡尔积)。
巧妙之处在于，我们可以将 [1,2] 构建不同左子树的数量表示为 G(2), 从 [4, 5, 6, 7]构建不同右子树的数量表示为 G(4)。这是由于 )G(n) 和序列的内容无关，只和序列的长度有关。于是，F(3,7)=G(2)⋅G(4)。 概括而言，我们可以得到以下公式：
F(i,n)=G(i−1)⋅G(n−i)  

### 代码  
```
class Solution {
public:
    int numTrees(int n) {
        // 动态规划
        vector<int> result(n+1);
        result[0] = 1;
        result[1] = 1;
        // 自底向上
        for(int i = 2; i <= n; ++i)
        {
            for(int j = 1; j <= i; ++j)
            {
                result[i] += result[j - 1] * result[i - j];
            }
        }
        return result[n];
    }
};
```
## 951. 翻转等价二叉树
### 题目  
我们可以为二叉树 T 定义一个翻转操作，如下所示：选择任意节点，然后交换它的左子树和右子树。

只要经过一定次数的翻转操作后，能使 X 等于 Y，我们就称二叉树 X 翻转等价于二叉树 Y。

编写一个判断两个二叉树是否是翻转等价的函数。这些树由根节点 root1 和 root2 给出。

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
    bool flipEquiv(TreeNode* root1, TreeNode* root2) {
        bool doflipequiv = false;
        if(root1 == NULL && root2 == NULL)
            return true;
        else if(root1 == NULL || root2 == NULL)
            return false;
        // 先判断根节点是否相等，如果不相等，就返回false
        if(root1->val != root2->val)
            return doflipequiv;
        // 假设没翻转
        doflipequiv = flipEquiv(root1->left, root2->left) && flipEquiv(root1->right, root2->right);
        // 假设翻转
        if(!doflipequiv)
            doflipequiv = flipEquiv(root1->left, root2->right) && flipEquiv(root1->right, root2->left);
        return doflipequiv;
        
    }
};
```
## 95. 不同的二叉搜索树 II
### 题目
给定一个整数 n，生成所有由 1 ... n 为节点所组成的 二叉搜索树 。

### 代码 
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        if(n <= 0) return {};
        return dogenerateTrees(1, n);
    }
    
    vector<TreeNode*> dogenerateTrees(int start, int end)
    {
        vector<TreeNode*> res;
        if(start > end)
        {
            res.push_back(nullptr);
            return res;
        }
        
        // 遍历
        for(int i = start; i <= end; ++i)
        {
            // 左子树
            auto leftlist = dogenerateTrees(start, i - 1);
            // 右子树
            auto rightlist = dogenerateTrees(i + 1, end);
           
            // 双遍历
            for(auto leftNode :leftlist) {
                for(auto rightNode:rightlist){
                     // 根节点
                    TreeNode* root = new TreeNode(i);
                    root->left = leftNode;
                    root->right = rightNode;
                    res.push_back(root);
                }
                    
            }
        }
        return res;
    }
};
```
## 1026. 节点与其祖先之间的最大差值
### 题目  
给定二叉树的根节点 root，找出存在于不同节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B 的祖先。

（如果 A 的任何子节点之一为 B，或者 A 的任何子节点是 B 的祖先，那么我们认为 A 是 B 的祖先）

### 题解  
就一个节点来说所谓最大差值，就是祖先的最大值或者最小值和自己的val的差值。
只需要知道所有祖先可能的最大值和最小值，在遍历时携带传递即可。

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
    int result = 0;
    int maxAncestorDiff(TreeNode* root) {
        // 最大差值就是该节点与父结点的最大值和最小值的差值
        if(root == NULL)
            return 0;
        dfs(root, root->val, root->val);
        return result;
    }

    void dfs(TreeNode* root, int max_befor, int min_before)
    {
        if(root == NULL)
            return;
        result = max(max(abs(root->val - max_befor), abs(root->val - min_before)), result);
        max_befor = max(root->val, max_befor);
        min_before = min(root->val, min_before);
        dfs(root->left, max_befor, min_before);
        dfs(root->right, max_befor, min_before);
    }

};
```
## 1457. 二叉树中的伪回文路径
### 题目  
给你一棵二叉树，每个节点的值为 1 到 9 。我们称二叉树中的一条路径是 「伪回文」的，当它满足：路径经过的所有节点值的排列中，存在一个回文序列。

请你返回从根到叶子节点的所有路径中 伪回文 路径的数目。

### 题解  
这题主要难点时如何确定路径为伪回文。
题中节点的值只能为1-9,充分利用这点。
我们用一个二进制数来维护判断是不是伪回文。
二进制数第一位的1，0表示值为1的节点奇偶性，第二位1，0表示值为2的节点奇偶性。。。

按 2,3,3路径来说。
2节点来的时候 temp 的二进制为00000010; temp^=1<<2;
3节点来的时候 temp 的二进制为00000110; temp^=1<<3;
3节点再来的时候 temp 的二进制为00000010; temp^=1<<3;
当到达叶子节点的时候，如果有偶数个元素,如果是伪回文，temp==0;
奇数个时，temp的二进制有一位为1.temp&(temp-1)==0

### 代码
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int result = 0;
    int pseudoPalindromicPaths (TreeNode* root) {
        if(root == NULL)
            return 0;
        dfs(root, 0);
        return result;
    }

    void dfs(TreeNode* root, int temp)
    {
        temp ^= (1 << root->val);
        if((root->left == NULL) && (root->right == NULL))
        {
            if(temp == 0 || ((temp - 1) & temp) == 0)
                result ++;
        }

        if(root->left != NULL) dfs(root->left, temp);
        if(root->right != NULL) dfs(root->right, temp);


    }
};
```
## 116. 填充每个节点的下一个右侧节点指针
### 题目
给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

### 题解

使用已建立的 next 指针
### 代码
```
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(NULL), right(NULL), next(NULL) {}

    Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};
*/

class Solution {
public:
    Node* connect(Node* root) {
        if(root == NULL)
            return root;
        // 最左边的节点
        Node* leftMost = root;
        
        // 当最左边的节点的左子树为NULL时，表示已经到达叶子了
        while(leftMost->left != NULL)
        {
            // 同一层
            Node* curNode = leftMost;
            while(curNode != NULL)
            {
                // 先把左子树和右子树相连
                curNode->left->next = curNode->right;
                // 再把右子树和隔壁节点的左子树相连
                if(curNode->next != NULL)
                    curNode->right->next = curNode->next->left;
                // 遍历next节点
                curNode = curNode->next;
            }
            // 下一层的最左边节点
            leftMost = leftMost->left;
            
        }

        return root;
    }
};
```
## 919. 完全二叉树插入器
### 题目  
完全二叉树是每一层（除最后一层外）都是完全填充（即，节点数达到最大）的，并且所有的节点都尽可能地集中在左侧。

设计一个用完全二叉树初始化的数据结构 CBTInserter，它支持以下几种操作：

CBTInserter(TreeNode root) 使用头节点为 root 的给定树初始化该数据结构；
CBTInserter.insert(int v)  向树中插入一个新节点，节点类型为 TreeNode，值为 v 。使树保持完全二叉树的状态，并返回插入的新节点的父节点的值；
CBTInserter.get_root() 将返回树的头节点。
 

示例 1：

输入：inputs = ["CBTInserter","insert","get_root"], inputs = [[[1]],[2],[]]
输出：[null,1,[1,2]]
示例 2：

输入：inputs = ["CBTInserter","insert","insert","get_root"], inputs = [[[1,2,3,4,5,6]],[7],[8],[]]
输出：[null,3,4,[1,2,3,4,5,6,7,8]]

### 题解
搞两个队列，宽度优先遍历，一个队列存所有的节点，一个队列存不满足的节点  
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
class CBTInserter {
public:
    // 添加一个辅助队列，将所有不满足的节点全部搞进去
    queue<TreeNode*> leaf_or_singlechild;
    TreeNode* _root;
    CBTInserter(TreeNode* root) {
        _root = root;
        queue<TreeNode*> allNode;
        if(root != NULL)
            allNode.push(root);
        while(!allNode.empty())
        {
            TreeNode* node = allNode.front();
            allNode.pop();
            if(node->left == NULL || node->right == NULL)
                leaf_or_singlechild.push(node);
            if(node->left != NULL)
                allNode.push(node->left);
            if(node->right != NULL)
                allNode.push(node->right);
        }
    }
    
    int insert(int v) {
        TreeNode* newNode = new TreeNode(v);
        bool is_empty = leaf_or_singlechild.empty();
        leaf_or_singlechild.push(newNode);
        if(!is_empty)
        {
            TreeNode* curNode = leaf_or_singlechild.front();
            if(curNode->left == NULL)
                curNode->left = newNode;
            else {
                curNode->right = newNode;
                leaf_or_singlechild.pop();
            }
            return  curNode->val;
        }
        return v;
    }
    
    TreeNode* get_root() {
        return _root;
    }
};

/**
 * Your CBTInserter object will be instantiated and called as such:
 * CBTInserter* obj = new CBTInserter(root);
 * int param_1 = obj->insert(v);
 * TreeNode* param_2 = obj->get_root();
 */
```

## 1110. 删点成林  
### 题目  
给出二叉树的根节点 root，树上每个节点都有一个不同的值。

如果节点值在 to_delete 中出现，我们就把该节点从树上删去，最后得到一个森林（一些不相交的树构成的集合）。

返回森林中的每棵树。你可以按任意顺序组织答案。

### 代码  
```
*/
class Solution {

    List<TreeNode> res;
    HashSet<Integer> set;

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        /*
        *首先定义一个哈希set，用来放所有想要删掉的值
        *好处：回头在递归的时候可以只遍历整棵树一遍就完成删点成林
        */
        set = new HashSet<>();
        for(int eachNum : to_delete) {
            set.add(eachNum);
        }

        res = new LinkedList<>();

        //这里传入的 needAdd 的值为true。意思是：接下来传入的点 有可能 被添加到最终结果res中
        helper(root, true);
        return res;
    }
    /*
    *这个helper函数很有意思，需要解释一下，只要明白这个函数的返回值，参数什么意思就理解了大半
    * @return:返回值可以判断这个 传入的点 是不是需要删除的点
    * @param needAdd :该参数判断这个 点 是否应该被 加入 最终结果内
    */
    private boolean helper(TreeNode node, boolean needAdd) {
        if(node == null) return false;

        //注意，如果set中有node.val的话，那么就意味着这个点需要删除
        if(set.contains(node.val)) {

            //这里依旧是传入参数 needAdd 为 true，意思同上：
            //接下来这个传入的节点 又可能 被添加入最终结果res中
            if(helper(node.left, true)) node.left = null;
            if(helper(node.right, true)) node.right = null;

            //直接返回，不做后续处理了，直接返回true，意思是这个点应该被删除
            return true;
        }

        //如果这个点是 期望中 需要添加的点，将之加入结果的线性表中
        if(needAdd == true) res.add(node);

        /*
        *仔细看,这里的needAdd是传入的false，意思是:下一个点 肯定 不会被加入最终结果res中
        *这里的 if 判断也有意思，如果判断通过的话 
        *意思是: helper 返回了true，即，这个helper传入的点需要删除
        *既然这个传入的点需要被删除，那么就从这里断开：node.left = null;  or  node.right = null;
        */
        if(helper(node.left, false)) node.left = null;
        if(helper(node.right, false)) node.right = null;
        return false;
    }
}
```  
## 面试题 04.06. 后继者
### 题目
设计一个算法，找出二叉搜索树中指定节点的“下一个”节点（也即中序后继）。

如果指定节点没有对应的“下一个”节点，则返回null。  
### 题解
所谓 p 的后继节点，就是这串升序数字中，比 p 大的下一个。

如果 p 大于当前节点的值，说明后继节点一定在 RightTree
如果 p 等于当前节点的值，说明后继节点一定在 RightTree
如果 p 小于当前节点的值，说明后继节点一定在 LeftTree 或自己就是
递归调用 LeftTree，如果是空的，说明当前节点就是答案
如果不是空的，则说明在 LeftTree 已经找到合适的答案，直接返回即可

### 代码
```
TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
        if (!root) {
            return NULL;
        }

        if (root->val <= p->val) {
            return inorderSuccessor(root->right, p);
        } else {
            TreeNode *leftRet = inorderSuccessor(root->left, p);
            if (!leftRet) {
                return root;
            } else {
                return leftRet;
            }
        }
    }
```
## 652. 寻找重复的子树
### 题目
给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意一棵的根结点即可。

两棵树重复是指它们具有相同的结构以及相同的结点值。

### 题解
本题需要寻找重复的子树，这里重复的子树是指具有相同的结构以及相同的结点值。我们需要进行序列化的操作。首先要介绍一下序列化：

序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

通过一个编码还有一个解码的方式来得到原数据，那么就从侧面说明，对于原数据来说，使用相同序列化后的结果肯定是唯一的。所以，既然我们要找重复的子树，那么我们只需要将所有子树都使用相同的方式进行序列化，那在此过程中如果发现有相同的序列，那我们就可以找出相同的子树。

至此，对于我们解二叉树的题无非就以下几种思路：

先序遍历（深度优先搜索）
中序遍历（深度优先搜索）（尤其二叉搜索树）
后序遍历（深度优先搜索）
层序遍历（广度优先搜索）
序列化与反序列化（结构唯一性问题）
序列化二叉树示意图：

### 代码
```
class Solution {
public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        vector<TreeNode*> res;
        unordered_map<string, int> mp;
        dfs(root, res, mp);
        return res;
    }
    
    string dfs(TreeNode* root, vector<TreeNode*>& res, unordered_map<string, int>& mp){
        if(root==0) return "";
        //二叉树先序序列化
        string str = to_string(root->val) + "," + dfs(root->left, res, mp) + "," + dfs(root->right, res, mp);
        
        if(mp[str]==1){
            res.push_back(root);
        } 
        mp[str]++;
        return str;
    }
};

```
## 面试题 04.05. 合法二叉搜索树
### 题目
实现一个函数，检查一棵二叉树是否为二叉搜索树。

### 题解
左子树只要小于根节点就好，右子树得大于根节点，小于根节点的根节点；
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
    bool isValidBST(TreeNode* root) {
        return doisValidBST(root, NULL, NULL);
    }

    bool doisValidBST(TreeNode* root, TreeNode* min, TreeNode* max)
    {
        if(root == NULL)
            return true;
        if(min != NULL && root->val <= min->val) return false;
        if(max != NULL && root->val >= max->val) return false;
        return doisValidBST(root->left, min, root) && doisValidBST(root->right, root, max);
    }
};
```
## 662. 二叉树最大宽度
### 题目
给定一个二叉树，编写一个函数来获取这个树的最大宽度。树的宽度是所有层中的最大宽度。这个二叉树与满二叉树（full binary tree）结构相同，但一些节点为空。

每一层的宽度被定义为两个端点（该层最左和最右的非空节点，两端点间的null节点也计入长度）之间的长度。

### 题解  
关键点在于这一层的长度是由这一层的最左侧节点与最右侧节点来计算。
与树中的值没有关系，请屏蔽里面的内容来解题。
假设取第二层，即 d=2 d = 2d=2 的某个结点。

满二叉树中，某个结点的左孩子节点位置在 2∗d 2 * d2∗d，右孩子节点位置在 2∗d+1 2*d+12∗d+1
想办法记录当前层其中一侧(最左侧)的节点所在位置，之后遇到当前层的其它节点时计算它们之间的距离。
这里将图片的数字理解成位置，比如 4~6 就是 6−4+1=3 6-4+1 = 36−4+1=3，即距离3

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
    int widthOfBinaryTree(TreeNode* root) {
        if(root == NULL)
            return 0;
        queue<pair<TreeNode*, unsigned long long>> q;
        q.push({root, 0});
        int result = 0;
        while(!q.empty())
        {
            int size = q.size();
            result = max(int(q.back().second - q.front().second + 1), result);
            for(int i = 0; i < size; ++i)
            {
                TreeNode* cur = q.front().first;
                unsigned long long pos = q.front().second;
                q.pop();
                if(cur->left) q.push({cur->left, 2 * pos});
                if(cur->right) q.push({cur->right, 2 * pos + 1});
            }
        }
        return result;
    }
};
```
## 863. 二叉树中所有距离为 K 的结点
### 题目
给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。

返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。

 

示例 1：

输入：root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2
输出：[7,4,1]
解释：
所求结点为与目标结点（值为 5）距离为 2 的结点，
值分别为 7，4，以及 1

### 题解
关键在于将子节点和父节点先存起来，然后宽度优先遍历，找到第K层
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
    unordered_map<TreeNode*, TreeNode*> parent_map;
    vector<int> distanceK(TreeNode* root, TreeNode* target, int K) {
        // 先把节点的父结点存进去
        dfs(root, NULL);
        vector<int> result;
        queue<TreeNode*> queue1;
        unordered_set<TreeNode*> visted;
        if(target != NULL) {
            queue1.push(target);
            visted.insert(target);
        }

        while(!queue1.empty())
        {
            if(K-- == 0)
            {
                while(!queue1.empty())
                {
                    result.push_back(queue1.front()->val);
                    queue1.pop();
                }
                return result;
            } else {
                int size = queue1.size();
                for(int i = 0; i < size; ++i)
                {
                    TreeNode* node = queue1.front();
                    queue1.pop();
                    if(node->left && visted.find(node->left) == visted.end())
                    {
                        queue1.push(node->left);
                        visted.insert(node->left);
                    }
                    if(node->right && visted.find(node->right) == visted.end())
                    {
                        queue1.push(node->right);
                        visted.insert(node->right);
                    }
                    if(parent_map[node] != NULL && visted.find(parent_map[node]) == visted.end())
                    {
                        queue1.push(parent_map[node]);
                        visted.insert(parent_map[node]);
                    }
                }
            }
        }

        return result;
            
    }

    void dfs(TreeNode* child, TreeNode* parent)
    {
        if(child != NULL)
        {
            parent_map.insert({child, parent});
            dfs(child->left, child);
            dfs(child->right, child);
        }
    }
};
```
## 718. 最长重复子数组
### 题目
给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

 

示例：

输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：
长度最长的公共子数组是 [3, 2, 1] 。

### 题解  
关键在于建立动态规划的思想  
dp[i][j]表示A[i:]和B[j:]最长的子串前缀：在此例中dp[0][0] = 0；dp[1][0] = 0; dp[2][0] = 3  

### 代码  
```
class Solution {
public:
    int findLength(vector<int>& A, vector<int>& B) {
        int A_size = A.size();
        int B_size = B.size();
        // dp[i][j]表示A[i:]和B[j:]最长的子串前缀：在此例中
        // dp[0][0] = 0；dp[1][0] = 0; dp[2][0] = 3
        vector<vector<int>> dp(A_size + 1, vector<int>(B_size + 1, 0));
        int ans = 0;
        // 从后往前遍历, 为啥从后往前遍历
        // 假设从前往后遍历
        // A = [1, 2, 3]; B = [1, 2, 3];
        // 则dp[0][0] = dp[1][1] + 1;而此时dp[1][1]并没有算出来
        // 从后往前遍历
        // dp[2][2] = dp[3][3] + 1 = 0 + 1;正好
        for(int i = A_size - 1; i >= 0; i --)
            for(int j = B_size - 1; j >= 0; j--)
            {
                dp[i][j] = (A[i] == B[j]) ? (dp[i + 1][j + 1] + 1) : 0;
                ans = max(ans, dp[i][j]);
            }
        return ans;
    }
};
```  
## 121. 买卖股票的最佳时机  
### 题目  
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。

注意：你不能在买入股票前卖出股票。

 

示例 1:

输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
示例 2:

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

### 代码
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if(n == 0)
            return 0;
        int cur_min = prices[0];
        int cur_max_value = 0;
        for(int i = 0; i < n; ++i)
        {
            // 如果当前节点小于之前的最小值，那就不用求利润了，因为利润为负
            if(prices[i] < cur_min)
                cur_min = prices[i];
            // 如果当前节点大于之前的最小值，且它们之间的差值大于之前的最大利润，就更新当前的最大利润
            else if(prices[i] - cur_min > cur_max_value)
                cur_max_value = prices[i] - cur_min;
        }
        return cur_max_value;
    }
};
``` 
## 53. 最大子序和
### 题目
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

### 题解  
假设 nums 数组的长度是 nn，下标从 0 到 n - 1。用 f(i) 代表以第 i 个数结尾的「连续子数组的最大和；
那么 f(i) = max(f(i - 1) + nums[i], nums[i])



### 代码 
```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int size = nums.size();
        if(size == 0)
            return 0;
        int pre = nums[0];
        int max_number = pre;
        for(int i = 1; i < size; ++i)
        {
            pre = max(pre + nums[i], nums[i]);
            max_number = max(pre, max_number);
        }
        return max_number;
    }
};
```

## 141. 环形链表
### 题目  
给定一个链表，判断链表中是否有环。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。

### 代码
```
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
    bool hasCycle(ListNode *head) {
        // 用双指针来搞，一个指针每次走两步，另一个指针每次走一步，如果他们相遇，则说明链表中有环
        if(head == NULL)
            return false;
        bool is_cycle = false;
        ListNode* two_step = head;
        ListNode* one_step = head;
        while(two_step != NULL && two_step->next != NULL)
        {
            two_step = two_step->next->next;
            one_step = one_step->next;
            if(two_step == one_step) {
                is_cycle = true;
                break;
            }  
        }

        return is_cycle;

    }
};
```
## 198. 打家劫舍
### 题目 
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

 

示例 1：

输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
示例 2：

输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。

### 代码
```
class Solution {
public:
    int rob(vector<int>& nums) {
        // 用动态规划来搞dp(i) = max(dp(i + 1), nums[i] + dp(i + 2));
        // dp(i)表示，在第i家开始，到之后能抢到最大金额。
        // 有两种状态，第一种不抢第i家，则金额等于从第i + 1家开始抢的金额
        // 第二种，抢第i家，则金额等于从第i + 2家开始抢的金额。
        
        int size = nums.size();
        if(size == 0)
            return 0;
        int max_money = 0;
        int dp_i_2 = 0;
        int dp_i_1 = 0;
        int dp_i = 0;
        // 从后往前递归
        for(int i = size - 1; i >= 0; i--)
        {
            dp_i = max(dp_i_1, nums[i] + dp_i_2);
            max_money = max(max_money, dp_i);
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return max_money;
    }
};
```
## 39. 组合总和  
### 题目  
给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 
示例 1:

输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]
示例 2:

输入: candidates = [2,3,5], target = 8,
所求解集为:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]

###  题解
用回溯法搞

### 代码
```
class Solution {
public:
    vector<vector<int>> paths;
    vector<int> path;
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        int size = candidates.size();
        if(size == 0)
            return paths;
        // 排序
        sort(candidates.begin(), candidates.end());
        dfs(candidates, target, 0);
        return paths;
    }

    void dfs(vector<int>& candidates, int target, int start)
    {
        // 边界
        if(target == 0)
        {  
            paths.push_back(path);
            return;
        }

        if(target < 0)
            return;

        for(int i = start; (i < candidates.size()) && (target - candidates[i] >= 0); i++)
        {
            // 加进去
            path.push_back(candidates[i]);
            // 递归
            dfs(candidates, target - candidates[i], i);
            // 撤回
            path.pop_back();
        }
        // 
    }
};
```
## 279. 完全平方数
### 题目  
给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

示例 1:

输入: n = 12
输出: 3 
解释: 12 = 4 + 4 + 4.
示例 2:

输入: n = 13
输出: 2
解释: 13 = 4 + 9.

### 代码
```
class Solution {
public:
    int numSquares(int n) {
        // 用动态规划来搞
        // 对于一个正整数N，所有的解都是N = 一个整数的平方 + 另一个数
        // 那公式就来了：dp(n) = min( 1(j的平方) + dp(n - j * j) ) , j * j < n
        int result[n + 1];
        result[0] = 0;
        for(int i = 1; i <= n; i ++)
        {
            result[i] = i; // 最坏的情况就是 1 + 1 + 1 ...
            for(int j = 1; (j * j) <= i; ++j)
                result[i] = min(result[i], (1 + result[i - j * j]));
        }

        return result[n];
    }
};
```

## 300. 最长上升子序列
### 题目  
给定一个无序的整数数组，找到其中最长上升子序列的长度。

示例:

输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。

### 代码  
```
class Solution {
public:
    // 动态规划来搞，
    // dp[i]表示以nums[i]为结尾的最长递增子序列长度
    // 以nums[i]为结尾的最长递增子序列中，nums[0,...,i-1]中所有比nums[i]小的都可以做为倒数第二个数，
    // 在这么多倒数第二个数的选择中，以哪个数结尾的最长递增子序列最长，就选哪个数为倒数第二个数，
    // 所以转移方程就出来了 dp[i] = max{dp[j] + 1}; 其中j < i; 且nums[j] < nums[i] 
    int lengthOfLIS(vector<int>& nums) {
        int length = nums.size();
        int res = 0;
        // 以nums[i]为结尾的递增子序列长度最短为1，所以先赋值dp[i] = 1
        vector<int> dp(length, 1);
        for(int i = 0; i < length; ++i)
        {
            for(int j = 0; j < i; ++j)
            {
                if(nums[i] > nums[j])
                    dp[i] = max(dp[i], dp[j] + 1);
            }

            res = dp[i] > res ? dp[i] : res;
        }

        return res;
    }
};
```

## 560. 和为K的子数组
### 题目  
给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。

示例 1 :

输入:nums = [1,1,1], k = 2
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。

### 代码  
```
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        // 用map来搞
        // 假设s[i] = nums[0] + nums[1] + ... + nums[i];
        // 那么连续的子数组nums[i] + nums[i + 1] + ... + nums[j] = k, 可以用s[j] - s[i - 1] = k来表示，这里把s[j]固定
        // 那么只需找到s[i - 1] = s[j] - k，用hash表记录下s[i - 1]的值出现的次数
        map<int, int> hash;
        hash[0] = 1;
        int temp = 0;
        int res = 0;
        for(auto i : nums)
        {
            temp += i;
            if(hash.find(temp - k) != hash.end()) res += hash[temp - k];
            hash[temp]++;
        }
        return res;
    }
};
```
