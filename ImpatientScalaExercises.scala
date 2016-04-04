def signum (a : Int): Int = if (a < 0) -1 else if (a == 0) 0 else 1

for (i <- 10 to (0,-1)) println(i)

def countdown (n: Int) = {
  for (i <- n to (0, -1)) println(i)
}

def unicodeProductString (s: String): BigInt = s.getBytes("UTF-8").foldLeft(1)(_*_)

def factorial(i: Int): Int = {
  (1 to i).foldLeft(1)(_*_)
}

def epowseries(x: Float, t: Int):Float= {
  val terms = (1 to t)
  val out = terms.map { i =>
    (Math.pow(x, i) / factorial(i)).toFloat
  }
  out.foldLeft(1f)((a: Float,b: Float) => (a+b))
}

def getRandomArray(n: Int): Array[Int] = for (i <- (0 until n)) yield scala.util.Random.nextInt(n)

def swapPairs(arr: Array[Int]): Array[Int] = {
  for (i <- (0 until (arr.length, 2))) {
    if (i != a.length - 1) {  
      val temp = a(i)
      a(i) = a(i+1)
      a(i+1) = temp
    }
  }
}
