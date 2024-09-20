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


## Graphs