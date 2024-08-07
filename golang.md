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
}

```