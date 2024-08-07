## Learning Golang

Learning from a Tour of Go: https://go.dev/tour/welcome/1

### Exercise: Loops and Functions
```golang
package main

import (
	"fmt"
)

func Sqrt(x float64) (z float64) {
	z = 1.0
	i := 10
	for i > 0 {
		z -= (z*z - x) / (2*z)
		i -= 1
	}
	return
}

func main() {
	fmt.Println(Sqrt(2))
}

```