object Solution {
    def main(args: Array[String]) {
        val stdin = io.Source.stdin.getLines().toList
        val nums = stdin(1).split(" ").map(_.toDouble)
        // 
        def mean(nums: Array[Double]): Double = nums.foldLeft(0.0)(_+_) / nums.length.toDouble
        //
        def median(nums: Array[Double]): Double = {
            val nl = nums.length
            val ns = nums.sorted
            if (nl%2 != 0) ns(nl/2 + 1) else mean( Array(ns(nl/2-1), ns(nl/2)) )
        }
        //
        def mode(nums: Array[Double]) = {
            // count occurences
            val occur = nums.groupBy(identity).map(x => (x._1, x._2.size)).toList
            // Filter by count, keep only the ones that scored == max
            val mostFrequent = occur.filter( x => (x._2 == occur.maxBy(_._2)._2) )

            if (mostFrequent.size > 1) nums.min else mostFrequent(0)._1
        }
        //
        def std(nums: Array[Double]) = {
            //
            val m = mean(nums)
            val ss = nums.map { x =>
                Math.pow(x - m, 2)
            }.foldLeft(0.0)(_+_)                
            Math.sqrt(ss/nums.length)            
        }
        val meanVal = mean(nums)
        val medianVal = median(nums)
        val modeVal = mode(nums).toInt
        val stdVal = std(nums)
        println(f"$meanVal%.1f")
        println(f"$medianVal%.1f")
        println(modeVal)
        println(f"$stdVal%.1f")
    }
}

