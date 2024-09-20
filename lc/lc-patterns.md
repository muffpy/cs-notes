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

## Two Pointers

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