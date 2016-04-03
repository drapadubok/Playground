object Solution {
    def main(args: Array[String]) {
        // merge zipped strings, useful trick about converting tuple to list
        val stdin = io.Source.stdin.getLines().toList
        val merged = stdin(0).zip(stdin(1)).map( x => List(x._1, x._2) ).flatten.mkString
        
        println(merged)
    }
}
