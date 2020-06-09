# ��
## ������46. �����ַ�����ַ���
����һ�����֣����ǰ������¹����������Ϊ�ַ�����0 ����� ��a�� ��1 ����� ��b����������11 ����� ��l����������25 ����� ��z����
һ�����ֿ����ж�����롣����ʵ��һ����������������һ�������ж����ֲ�ͬ�ķ��뷽����
ʾ�� 1:

����: 12258
���: 5
����: 12258��5�ֲ�ͬ�ķ��룬�ֱ���"bccfi", "bwfi", "bczi", "mcfi"��"mzi"

���
�ö�̬�滮��˼�����⣺
�ȹ��ɣ�
12258���Ӻ���ǰ��ʼ����8����Ϊ���������룬���Է����8��58����Ȼ58�ǲ�����ģ�����������㣩����ô��������ֻ��Ҫ����1225��122���ͰѸղŵ�˼·����һ�Ρ�
������Ҫ�ж�����λ���Ƿ���[10, 25]�ķ�Χ�ڡ�

��������
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
		// ���numΪ��λ��ʱ
		if (num < 10) {
			count++;
			return;
		}

		// ȡ�������ڶ�λ�͵�����һλ
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

## ������ 04.03. �ض���Ƚڵ�����  
����һ�ö����������һ���㷨����������ĳһ��������нڵ���������磬��һ���������Ϊ D����ᴴ���� D ������������һ������������ȵ���������顣  
ʾ����  
![ʾ��](C:\picture_tmp\leetcode.PNG)  
��⣺  
���ý�ָoffer����Ľⷨ��������һ����������  
����q��[root]
��һ��ѭ��ʱ������1��
&emsp; ������root��Ȼ��root�������ӽ��Ž�ȥ����ʱq��[2, 3]  
�ڶ���ѭ��ʱ����������  
&emsp; ������2����2���ӽ�������к��棬��ʱq��[3,4,5]   
&emsp; ������3����3���ӽ�������к��棬��ʱq��[4,5,7]  
���ŵ����Σ����Ĵ�ѭ��  
��������:  
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
        // ���ö����Ƚ��ȳ�������
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
                // ��������ͷ
                {
                    result.push_back(node);
                    tmp = node;
                } else {
                    // ��������
                    tmp->next = node;
                    // �������һ�����
                    tmp = node;
                }

                // ���ӽڵ㱣��
                if(tree_tmp->left) q.push(tree_tmp->left);
                if(tree_tmp->right) q.push(tree_tmp->right);                   
            }
        
        }

        return result;
    }
};
```
