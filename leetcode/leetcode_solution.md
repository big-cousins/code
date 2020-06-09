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