### Problem #s
1, 3, 15, 20, 21, 53, 57, 102, 110, 121, 
125, 133, 141, 150, 200, 207, 209, 226, 232, 235, 
242, 278, 383, 409, 542, 704, 721, 733, 973, 994


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

### [383. Ransom Note](https://leetcode.com/problems/ransom-note/description/)
- use hashmaps to find letter frequecny in magazine
- for each letter found in ransom note, decrement frequency in hash map
- if a letter has neqgative frequency then note cannot be created from magazine

```cpp
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        if (ransomNote.size() > magazine.size()) return false;

        int freq[26] = { };
        for (auto ch: magazine) {
            freq[ch - 'a']++;
        }

        for (auto ch: ransomNote) {
            if (--freq[ch - 'a'] < 0) return false;
        }

        return true;
    }
};
```


## Prefix Sum

### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # create running products from left and right, multiply these values
        # left = nums[0] * nums[1] * .. nums[i-1]
        # right = nums[i+1] * ... nums[n-1]
        # answer[i] = left * right

        n = len(nums)
        l = nums[0]
        r = nums[-1]
        answer = [1] * n

        for i in range(1, n):
            answer[i] = l
            l *= nums[i]
        
        for i in range(n - 2, -1, -1):
            answer[i] *= r
            r *= nums[i]
        
        # print(answer)
        return answer
```

### [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/description/)

- we keep cumulative/prefix sum of zeroes and ones 
- we decrement 1 for zero, we increment 1 for one
- we store prefix sum state for every index
- we start with {0:-1} which means we saw prefix sum = 0 at index = -1
- when we process nums[i], if count += nums[i] was seen before, then we found a subarray of equal 0s and 1s of length (i - seen[count])

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        if not n: return 0

        seen = {0: -1}
        maxCount = 0
        count = 0
        for i, num in enumerate(nums):
            if num:
                count += 1
            else:
                count -= 1
            
            if count in seen:
                maxCount = max(maxCount, i - seen[count])
            else:
                seen[count] = i
        return maxCount
```

### [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/description/)

- use cached prefix sums to solve problem
- we store prefSum:count pairs
- we initliase with prefSums[0] = 1 which means prefSum of 0 was seen 1 time. Trivial because
subarray of size 0 has sum = 0.
- if (current prefix sum - k) was seen before, then we found a subarray of sum = k 

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        total = 0
        pref = 0
        prefSums = {}
        prefSums[0] = 1

        for num in nums:
            pref += num
            total += prefSums.get(pref - k, 0)
            prefSums[pref] = prefSums.get(pref, 0) + 1
        
        return total
```

### [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/description/)

- we record pref = (prefix sum) modulo k for each num in nums
- prefMods[] counts the number of times pref was seen before
- if sum[0, i] % K == sum[0, j] % K, sum[i + 1, j] is divisible by K
- so, we add all subarrays with the same value using prefMods[]

```python
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        res = 0
        pref = 0
        prefMods = [1] + [0] * k # 0, 1, 2, ... k
        for num in nums:
            pref = (pref + num) % k
            res += prefMods[pref]
            prefMods[pref] += 1
        return res
```

## Overlapping Intervals

### [57. Insert Interval](https://leetcode.com/problems/insert-interval/description/)

- we have newInterval = [s,e]
- all the items in intervals array which lie below s are collected in left[]
- all the items in intervals array which lie above e are collected in right[]
- overlapping intervals are merged

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        s, e = newInterval[0], newInterval[1]
        left = []
        right = []
        for i in intervals:
            if i[1] < s:
                left += [i]
            elif i[0] > e:
                right += [i]
            else:
                s = min(s, i[0])
                e = max(e, i[1])
        return left + [[s, e]] + right
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

### [15. 3Sum](https://leetcode.com/problems/3sum/description/)

straightword two-pointer approach:
- fix a number, find 2sum of (-1*number) from remaining numbers

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # Sort nums
        s_nums = sorted(nums)

        # Start from smallest negative integer, fix this value and find two sum
        if (s_nums[0] > 0): return []
        
        all_triples = set()
        for idx, num in enumerate(s_nums):
            # stop when we reach non-negative integers
            if num > 0: break

            # two-sum target
            target = 0 - num

            left = idx+1
            right = len(s_nums) - 1
            while left < right:
                sum = s_nums[left] + s_nums[right]
                if sum == target:
                    all_triples.add(tuple([num, s_nums[left], s_nums[right]]))
                    left += 1
                elif sum < target:
                    left += 1
                else:
                    right -= 1
        
        return all_triples
```

alternative approach:
- note the different types of triplet sums possible:
    - [-x, 0, x] = 0; if there is at least one zero in the list, we can create these triples for every (-x, x) pair in the list
    - [0, 0, 0] = 0; if there are at least 3 zeros in the list
    - [-x, -y, z] where -x + -y + z = 0 and x, y < 0 < z
    - [-x, y, z] where -x + y + z = 0 and x < 0 < y, z

- https://leetcode.com/problems/3sum/solutions/725950/python-5-easy-steps-beats-97-4-annotated/


## Sliding Window

### [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/)

- sliding window approach needed
- assume sum(nums[l:r]) >= target, we shift(/slide) l to l+k where sum(nums[l+k+1:r]) < target.
- compare r-l+1 to the minimum subarray size found so far

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

### [3. Longest Substring Without Repeating Characters] (https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

- assume s[l:r-1]is the longest substring without repeating chars and we are now processing s[r]
- assume we have a dictionary seen{} which records the last index we saw any charcter at.
- if s[r] is a repeating char and 
    - s[k] (where l <=  k < r) is where the character last occurs, then we need to start looking for a longest substring from s[k+1:].
    - we record that we've seen s[r] at position r
- if s[r] is not a repeating char then
    - we record we've seen s[r] at position r
    - we check if s[l:r] is the longest substring without repeating chars

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = {} # where (index) did we last see a character
        l = 0
        maxsub = 0
        for r, ch in enumerate(s):
            if ch in seen and seen[ch] >= l:
                l = seen[ch] + 1
            seen[ch] = r
            maxsub = max(r-l+1, maxsub)
        return maxsub
```

## Strings

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

### [409. Longest Palindrome](https://leetcode.com/problems/longest-palindrome/description/)

- get frequency map of letters in word
- observe that odd count letters can be used after decremting by 1
- observe that even count letters can be used without change
- if odd count letters occured, then we can use one of them without change

Version 1:

```cpp
class Solution {
public:
    int longestPalindrome(string s) {
        int freqs[58] = { };
        for (auto ch: s) {
            freqs[ch - 'A']++;
        }

        int total = 0; 

        for (int i = 0; i < 58; ++i) {
            int l = freqs[i];
            if (l % 2) total += (l-1);
            else total += l;
        }
        
        if (total < s.length()) return total + 1; // odd counts exist
        else return total;
    }
};
```

Version 2 (smaller):
```cpp
int longestPalindrome(string s) {
    int odds = 0;
    for (char c='A'; c<='z'; c++)
        odds += count(s.begin(), s.end(), c) & 1;
    return s.size() - odds + (odds > 0);
}
```

## Binary Search

### [704. Binary Search](https://leetcode.com/problems/binary-search/description/)

- get midpoint using (r+l)//2
- Python floor division op (//) rounds to the nearest integer
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l <= r:
            m = (r + l) >> 1
            if (nums[m] > target):
                r = m - 1
            elif (nums[m] < target):
                l = m + 1
            else:
                return m
        return -1
```

### [278. First Bad Version] (https://leetcode.com/problems/first-bad-version/description/)

- binary search until left == right

```python
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left = 1
        right = n
        while left < right:
            mid = (right + left) // 2
            if not isBadVersion(mid):
                left = mid+1
            else:
                right = mid
        return left
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

### [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/description/)

- input stack hold elements in LIFO order, output stack holds the same elements (from the input stack) in FIFO order
- push(x) has time complexity 0(1). we also store the first element inserted for peeking.
- pop(x) has amortized time complexity O(1) and worst-case time complexity O(n) when the output stack is empty.
- peek(x) has time complexity O(1) as we either front stored from before or we can get the top elemnt of output stack.

```cpp
#include <stack>

class MyQueue {
public:
    std::stack<int> input;
    std::stack<int> output;
    int front;

    MyQueue() {
        
    }
    
    void push(int x) {
        if (input.empty()) {
            front = x;
        }
        input.push(x);
    }
    
    int pop() {
        if (output.empty()) {
            while (!input.empty()) {
                int top = input.top();
                output.push(top);
                input.pop();
            }
        }
        int top = output.top();
        output.pop();
        return top;
    }
    
    int peek() {
        if (!output.empty()){
            return output.top();
        }
        return front;
    }
    
    bool empty() {
        return input.empty() && output.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

### [150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/description/)

- use stack to evaluate rpn
- if token is an operator, pop twice to get operands, eval expr, add to stack

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        res = 0

        for t in tokens:
            if t in "+-/*": # eval
                r, l = stack.pop(), stack.pop()
                if t == "+":
                    stack.append(l+r)
                elif t == "-":
                    stack.append(l-r)
                elif t == "*":
                    stack.append(l*r)
                else:
                    stack.append(int(l/r))
            else:
                stack.append(int(t))
        
        return stack.pop()
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

### [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)

while root exists, check:
- if root is greater than p and q, then try LCA(root.left)
- if root is smaller than p and q, then try LCA(root.right)
- if root is in between p and q, then we're done

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if (root.val > p.val and root.val > q.val):
                root = root.left
            elif (root.val < p.val and root.val < q.val):
                root = root.right     
            else:
                return root
```

### [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/description/)

- the naive (N^2) solution : abs(depth(left), depth(right)) <= 1 and isBalanced(left) and isBalanced(right)
- if we could check for balance while recursing through the tree, we can improve the running time
- dfs from root on left and right subtrees
    - return 0 if root does not exist
    - return -1 if unbalanced height of subtrees
    - return n>0 if balanced and n = height of tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # dfs_height(root) has three possible return values which have 
        # different meanings
        #        0: height = 0 (node does not exist)
        #       -1: unbalanced tree
        #  {n > 0}: height of input root
        #
        def dfs_height(root):
            if root is None: return 0

            left = dfs_height(root.left)
            if (left == -1): return -1
            right = dfs_height(root.right)
            if (right == -1): return -1

            if (abs(left - right) > 1): return -1
            return max(left, right) + 1
            
        return dfs_height(root) > -1
```

### [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)

- bfs using deque for each level of tree
- collect values of nodes for each, then append the leaf nodes to the queue
- collect each level in res[]

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []

        q = deque()
        q.append(root)

        while q:
            l_nodes = len(q) # number of nodes in this level
            level = []
            for i in range(l_nodes):
                node = q.popleft()
                if node:
                    level.append(node.val)
                    if node.left: q.append(node.left)
                    if node.right: q.append(node.right)
            if level:
                res += [level]
        return res
```

## Graphs

### [733. Flood Fill](https://leetcode.com/problems/flood-fill/description/)

- dfs outward from starting point
- if cell has required color, then color the cell and continue dfs
- else if cell is not required color or we reached end of grid, then return

```cpp
class Solution {
public:
    int r_size; int n_size;

    void fillr(vector<vector<int>>& image, int r, int c, int oldcolor, int newcolor) {
        if (r < 0 || c < 0 || r == this->r_size || c == this->n_size) {
            return;
        }
        if (image[r][c] != oldcolor) {
            return;
        }

        image[r][c] = newcolor;
        fillr(image, r-1,c,oldcolor,newcolor);
        fillr(image, r+1,c,oldcolor,newcolor);
        fillr(image, r,c-1,oldcolor,newcolor);
        fillr(image, r,c+1,oldcolor,newcolor);
    }

    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newcolor) {
        this->r_size = image.size();
        this->n_size = image[0].size();
        int oldcolor = image[sr][sc];
        if (oldcolor != newcolor) {
            fillr(image, sr, sc, oldcolor, newcolor);
        }
        return image;
    }
};
```

### [133. Clone Graph](https://leetcode.com/problems/clone-graph/description/)
DFS
- we have a hashmap (visited{}) mapping old nodes to newly cloned nodes
- assume we have a clone() function which clones nodes if they have not been cloned yet (not found in visited{}) and then clone()'s its neighbors
- if we define this function, we're done

BFS
- same as dfs, keep visited{}, level-wise traversal, clone if neighbor node is visited first time, then append to neighbor list

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return
        
        #bfs
        copy = Node(node.val, [])
        visited = {}
        visited[node] = copy

        q = deque()
        q.append(node)
        while q:
            node = q.popleft()
            for nei in node.neighbors:
                if nei not in visited:
                    ncopy = Node(nei.val, [])
                    visited[nei] = ncopy
                    q.append(nei)
                visited[node].neighbors.append(visited[nei])
        
        return copy

        #dfs
        visited = {}
        def dfs(node):
            if node in visited:
                return visited[node]
            else:
                copy = Node(node.val, [])
                visited[node] = copy
                for nei in node.neighbors:
                    copy.neighbors.append(dfs(nei))
                return copy
        return dfs(node) if node else None
```

### [207. Course Schedule](https://leetcode.com/problems/course-schedule/description/)

Topological Sort/Cycle finding/DAG verify

DFS
- graph[[]] stores the adjacency list of each course
    - graph[i] = [a,b,c] means course i is a prereq for courses a, b, c 
- visited[] can 3 values for each node (course)
    - 0 : unvisited
    - -1: temporary visit
    - 1 : permenant visit
- we recursively dfs from node i, create temporary mark and then dfs through neighbors reachable from i
- if we encounter i again with a temporary mark, we found a cycle. we're done
- if we finish the search from i, then create permenant mark and continue to next node
- if we permenenantly mark every node (visited = [1,1,...1]) , then we're done

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        pre = prerequisites
        graph = [[] for i in range(numCourses)]
        degree = [0] * numCourses #bfs
        for x,y in pre:
            graph[x].append(y)
            degree[y] += 1

        visited = [0] * numCourses
        #dfs
        # -1: temporary mark
        #  1: permenant mark
        #  0: no mark
        def dfs(i):
            if visited[i] == -1:
                return False
            if visited[i] == 1:
                return True
            
            visited[i] = -1 # add temp mark
            out = graph[i]
            for o in out:
                if not dfs(o):
                    return False
            visited[i] = 1
            return True
            
        for i in range(numCourses):
            if not dfs(i): 
                return False

        return True
```

### [721. Accounts Merge](https://leetcode.com/problems/accounts-merge/description/)

- let account index in accounts[] represent a unique id
- create mapping from emails to accounts
    - example: {'johnsmith@mail.com': [0, 1], 'john_newyork@mail.com': [0], 'john00@mail.com': [1], 'mary@mail.com': [2], 'johnnybravo@mail.com': [3]}
- for every account email, find the neighbor account mappings and merge emails from these accounts using dfs
- collect these merged e-mails in set():merged_emails
- create new account info using merged emails and name

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        # let id of account = index in accounts[]

        # create mapping from e-mails to accounts
        email_to_acc = {}
        for i in range(len(accounts)):
            for email in accounts[i][1:]:
                email_to_acc.setdefault(email,[]).append(i)
        
        # print(email_to_acc)

        def dfs_merge(i, emails):
            if visited[i]:
                return
            visited[i] = 1
            for email in accounts[i][1:]:
                emails.add(email)
                for acc in email_to_acc[email]:
                    dfs_merge(acc, emails)
        
        # iterate through emails in accounts[] and find neighbor account mappings
        # collect emails from these neighbor accounts using dfs        
        new_accounts = []
        visited = [0] * len(accounts)
        for i in range(len(accounts)):
            if visited[i]: 
                continue
            merged_emails = set()
            dfs_merge(i, merged_emails)
            merged_details = [accounts[i][0]] + sorted(merged_emails)
            new_accounts += [merged_details]

        return new_accounts
```

### [79. Word Search](https://leetcode.com/problems/word-search/)

- straightforward dfs
- visited nodes need to be temporarily marked and then unmarked after dfs returns

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:

        def dfs(i, j, slice):
            if not slice:
                return True
            res = False
            if (0 <= i < len(board)) and (0 <= j < len(board[i])) and (slice[0] == board[i][j]):
                board[i][j] = '#'
                res = dfs(i+1,j,slice[1:]) or dfs(i-1,j,slice[1:]) or \
                        dfs(i,j+1,slice[1:]) or dfs(i,j-1,slice[1:])
                board[i][j] = slice[0]
            return res

        # find starting letter occurences
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0] and dfs(i,j, word):
                    return True
        return False
```

## Matrix

### [542. 01 Matrix](https://leetcode.com/problems/01-matrix/description/)

- we first mark non-zero cells as unprocessed (-1)
- we process the cells in topological order based on distance from zero-cell:
queue.pop(0)
- so, zero-cells are processed first. one-cells are processed next, and so on.
- in the end, we have minimum distance of all cells

Alternative solution (DP; check discussion section for images):
- observe recursive nature of problem: 
    - to compute mat[r][c], we need min(top,left,bottom,right) + 1
    - this assumes all 4 direction cells have been computed but that's impossible to know
- So, we first compute distances based on top and left neighbors in the first pass: 
mat[r][c] = min(top, left) + 1
- then we update distances based on bottom and right neighbors in second pass:
mat[r][c] = min(mat[r][c], bottom + 1, right + 1)

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        # find zeros, mark non-zeros as -1
        m, n = len(mat), len(mat[0])

        q = []
        for i in range(0,m):
            for j in range(0,n):
                if mat[i][j] == 0:
                    q.append((i,j))
                else:
                    mat[i][j] = -1

        # bfs outward from zeros in all 4 directions
        # unvisited nodes are marked -1
        # visited nodes have distance set from nearest zero
        # (r,c) = (row, col)
        # (dr, dc) = (row+dir, col+dir)
        dirs = [[0,1], [1,0], [0,-1], [-1,0]]
        while q:
            r, c = q.pop(0)
            for dir in dirs:
                dr, dc = r+dir[0], c+dir[1]
                if dr < 0 or dr == m or dc < 0 or dc == n or mat[dr][dc] != -1: continue
                mat[dr][dc] = mat[r][c] + 1
                q.append((dr,dc))

        return mat
```

### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
- iterate over grid, if we find land, then mark all reachable from here as visited (*)
- count this island and continue

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # mark all land reachable from (i,j) as 0
        def mark_dfs(i, j):
            if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
                return

            grid[i][j] = "*"
            mark_dfs(i+1, j)
            mark_dfs(i, j+1)
            mark_dfs(i-1, j)
            mark_dfs(i, j-1)

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (grid[i][j] == "1"):
                    mark_dfs(i, j) # found land
                    count += 1
        return count
```

### [994. Rotting Oranges] (https://leetcode.com/problems/rotting-oranges/description/)

- find all rotting oranges and count fresh oranges
- bfs outward from rotting ornages (in the current level/minute) and spread rot
- if fresh oranges left, return -1
- if no fresh oranges, return minutes

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        q = deque()
        fresh = 0
                
        # find all rotting and count fresh oranges
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (grid[i][j] == 2):
                    q.append((i,j))
                elif (grid[i][j] == 1):
                    fresh += 1
        
        minutes = 0
        offsets = [0, 1, 0, -1, 0]
        while q and fresh > 0:
            minutes += 1
            curlevel = len(q)
            for k in range(curlevel):
                x, y = q.popleft() # rot x, rot y
                for d in range(4):
                    dx, dy = x+offsets[d], y+offsets[d+1]
                    if (0 <= dx < len(grid) and 0 <= dy < len(grid[i]) and grid[dx][dy] == 1):
                        # spread rot
                        grid[dx][dy] = 2
                        q.append((dx,dy))
                        fresh -= 1

        return minutes if fresh == 0 else -1
```

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

### [141. Linked List Cycle] (https://leetcode.com/problems/linked-list-cycle/description/)
- tortoise and hare algorithm/ floyd's cycle detection algorithm
- https://ivanyu.me/blog/2013/11/24/finding-a-cycle-in-a-linked-list/

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast: return True
        return False
```

## Heap (Top 'K' elements)

### [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/description/)

- heapq is implemented as a min heap
- We keep a max heap of size K (using negative euclidean distances)
- For each item, we insert an item to our heap.
- If inserting an item makes heap size larger than k, then we immediately pop the smallest (but technically the largest distance) item after inserting ( heappushpop ).


```python
import heapq

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []

        for x,y in points:
            dist = -((x*x) + (y*y))
            if len(heap) == k:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))

        return [(x,y) for (_, x, y) in heap]
```



## Backtracking/Recursion

## Dynamic Programming

### [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/description/)

- assume we have a running sum and max subarray for nums[:i] and we are now processing nums[i]
- if our running sum for nums[:i] is negative, we can start a new running sum from nums[i]
- else, we can add nums[i] to the running sum
- we update max subarray if our new running sum is greater than current max subarray sum

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dpsum, maxSum = nums[0], nums[0]
        for i in range(1, len(nums)):
            dpsum = (dpsum + nums[i]) if dpsum > 0 else nums[i]
            maxSum = max(maxSum, dpsum)
        return maxSum
```

## Binary

## Math