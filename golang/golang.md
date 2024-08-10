## Learning Golang

Learning from a Tour of Go: https://go.dev/tour/welcome/1

### Exercise: Loops and Functions
```golang
package main

import (
	"fmt"
	"math"
)

func Sqrt(x float64) (i int, z float64) {
	z = 1.0
	i = 0
	for {
		err := (z*z - x)
		z -= err / (2*z)
		fmt.Println(err, z)
		
		if math.Abs(err) < 0.001 { break }
		
		i += 1
	}
	return
}

func main() {
	iters, ans := Sqrt(2)
	fmt.Println(iters, ans)
	
	mathlib_ans := math.Sqrt(2)
	diff := math.Abs(mathlib_ans - ans)
	fmt.Println(diff)
}

```

### Arrays and slices

```golang
package main

import "fmt"

func append2Slices() {
    var s []int
	printSlice(s)

	// append works on nil slices.
	s = append(s, 0)
	printSlice(s)

	// The slice grows as needed.
	s = append(s, 1)
	printSlice(s)

	// We can add more than one element at a time.
	s = append(s, 2, 3, 4)
	printSlice(s)
}

func makeSlices() {
	a := make([]int, 5)
	printSlice("a", a)

	b := make([]int, 0, 5)
	printSlice("b", b)

	c := b[:2]
	printSlice("c", c)

	d := c[2:5]
	printSlice("d", d)
}

func printSlice(s string, x []int) {
	fmt.Printf("%s len=%d cap=%d %v\n",
		s, len(x), cap(x), x)
}

func main() {
    makeSlices()
    append2Slices()
}

```

### Ranges
```go
package main

import "fmt"

func main() {
	pow := make([]int, 10)
	for i := range pow {
		pow[i] = 1 << uint(i) // == 2**i
	}
	for _, value := range pow {
		fmt.Printf("%d\n", value)
	}
}
```

### Exercise: Slices

```go
package main

import "golang.org/x/tour/pic"

func Pic(dx, dy int) [][]uint8 {
    
    out := make([][]uint8, dy)
	
	for i := range out {
	
		out[i] = make([]uint8, dx)
		
		for j := 0; j < dx; j++ {
			z := uint8( i * j )
			out[i][j] = z
		}
	}
	
	return out
}

func main() {
	pic.Show(Pic)
}

```
![alt text](image-x*y.png)