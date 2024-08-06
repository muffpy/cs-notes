## Learning Golang

Learning from a Tour of Go: https://go.dev/tour/welcome/1

### Exercise: Loops and Functions
```golang
package main

import (
	"fmt"
)

func Sqrt(x float64) float64 {
	var z float64 = 1.0
	z -= (z*z - x) / (2*z)
}

func main() {
	fmt.Println(Sqrt(2))
}

```