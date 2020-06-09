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
![示例](C:\picture_tmp\leetcode.PNG)  
题解：  
利用剑指offer里面的解法，即利用一个队列来解  
队列q：[root]
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
