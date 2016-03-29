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

