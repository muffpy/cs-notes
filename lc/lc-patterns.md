## Arrays & Hashing

### [1. Two Sum](https://leetcode.com/problems/two-sum/description/)
- store target-nums[k] for nums[0..i-1] (where 0 <= k <= i-1) in a dictionary
- if target - nums[i] exists in dictionary, then we found our (unique) solution
- else, store target-nums[i] in dictionary and continue

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        diffs = {}
        for idx, num in enumerate(nums):
            diff = target - num
            if diff in diffs:
                return [idx, diffs[diff]]
            diffs[num] = idx
        return []
```

### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)

- assume we have maxProfit(prices[:i-1]) and we also which day was the lowest price (minDay)
- if prices[i] - minDay > maxProfit(prices[:i-1]), then we have a new maxProfit (sell day)
- if prices[i] < minDay, then we have a new lowest price (buy day)
- else, we're done

```python
class Solution:
    # Beats 93% of solutions
    def maxProfit(self, prices: List[int]) -> int:
        minDay = prices[0]
        maxProfit = 0
        for i in prices[1:]:
            if i < minDay:
                minDay = i
                continue
            profit = i - minDay 
            if profit > maxProfit:
                maxProfit = profit

        return maxProfit
```

### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)
- ord() function in Python is **used to convert a single Unicode character into its integer representation.**
- create two arrays (for two strings) of length 26
- convert each letter in both strings to integer using ord() and scale it down to 0-26
- use this integer as an index and increment array value
- both arrays should be equal for anagrams

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s_arr, t_arr = [0]*26, [0]*26
        for i in t:
            t_arr[ord(i)-97] += 1
        for i in s:
            s_arr[ord(i)-97] += 1

        return s_arr == t_arr
```

## Two Pointers

### [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/description/)

- two-pointer approach from both ends of string
- first check if pointer values are alphanumeric and then check if equal

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l = 0
        r = len(s) - 1
        while l < r:
            if not s[l].isalnum():
                l += 1
            elif not s[r].isalnum():
                r -= 1
            elif s[r].lower() == s[l].lower():
                l += 1
                r -= 1
            else:
                return False
        return True
```

## Sliding Window

### [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/)

- sliding window approach needed
- assume sum(nums[l:r]) >= target, we shift(/slide) l to l+k where sum(nums[l+k+1:r]) < target.
- compare r-(l+k) to the minimum subarray size found so far

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        numsl = len(nums)
        l = 0
        lrsum = 0
        minsub = numsl + 1
        for r, num in enumerate(nums):
            lrsum += num
            if (lrsum >= target):
                while l != r and (newlr := lrsum - nums[l]) >= target:
                    lrsum = newlr
                    l += 1
                minsub = min(r-l+1, minsub)
        return 0 if minsub > numsl else minsub
```

## Stack

### [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/)
- simple stack problem
- iterate through string
    - if ch is a left bracket, add to stack
    - if ch is a right bracket and the stack is empty, return false
    - if ch is a right bracket and does not match the last ch in stack, return false
- if stack is empty in the end, then string is valid
```python
class Solution:
    def isValid(self, s: str) -> bool:
        st = []
        brpairs = {'(':')', '{':'}', '[':']'}
        for i in s:
            if i in brpairs:
                st.append(i)
            elif not st or (brpairs[st.pop()] != i):
                return False
        return st == []
```

## Binary Tree

### [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/)

- if root is null return null
- store the old right subtree
- invert the old left subtree and assign to new right subtree
- invert the (stored) old right subtree and assign to new left subtree
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root

        prevright = root.right
        root.right = self.invertTree(root.left)
        root.left = self.invertTree(prevright)

        return root
```

## Graphs

## Linked Lists

### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/description/)

- two pointers pointing to each linked list
- create new linked list using the smaller of two pointer values
- once you run out of a list, attach the remaining list to the end of new list

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p1 = list1
        p2 = list2
        p3 = ListNode()
        dummy = p3
        while p1 and p2:
            newnode = ListNode()
            if p1.val <= p2.val:
                newnode.val = p1.val
                p1 = p1.next
            else:
                newnode.val = p2.val
                p2 = p2.next
            p3.next = newnode
            p3 = newnode
        
        p3.next = p1 if p1 else p2
        newlist = dummy.next
        return newlist
```