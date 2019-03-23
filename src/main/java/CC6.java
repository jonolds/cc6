import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class CC6 extends MRHelp implements Tool {
	String log = ""; int maxIters = 4, mp = 3, rd = 3;
	/** WHITE and BLACK nodes are emitted as is. For every edge of a GRAY node, we emit a new Node with 
	 * distance incremented by one. The Color.GRAY node is then colored black and is also emitted. */
	public static class MapClass extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
			Node node = new Node(value.toString());

			// For each GRAY node, emit each of the edges as a new node (also GRAY)
			if (node.getColor() == Color.GRAY) {
				for(int i = 0; i < node.getEdges().size(); i++) {
					Node dummynode = new Node(node.getEdges().get(i));
					dummynode.setCost(node.getWeights().get(i) + node.getCost());
					dummynode.setColor(Color.GRAY);
					output.collect(new IntWritable(dummynode.getId()), dummynode.getLine());
				}
				node.setColor(Color.BLACK);
			}
			output.collect(new IntWritable(node.getId()), node.getLine());
		}
	}

	/** A reducer class that just emits the sum of the input values. */
	public static class Reduce extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {

		/** Make a new node which combines all information for this single node id. The Node should have
		 * - 1)The full list of edges. 2)The minimum distance. 3)The darkest Color. */
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
			List<String> vals = itToList(values);
			
			List<Integer> edges = null, weights = null;
			int cost = Integer.MAX_VALUE;
			Color color = Color.WHITE;

			print("\t\t\t\t\t\t\t" + key);
			for(int i = 0; i < vals.size(); i++) {
				println(i==0?("\t" + vals.get(i)) : ("\t\t\t\t\t\t\t" + "\t" + vals.get(i)));
				Node u = new Node(key.get() + "\t" + vals.get(i));

				if(u.getEdges().size() > 0)
					edges = u.getEdges();
				if(u.getWeights().size() > 0)
					weights = u.getWeights();
				// Save the minimum cost and darkest color
				if(u.getCost() < cost)
					cost = u.getCost();
				color = u.getColor();
//				color = (u.getColor().ordinal() > color.ordinal()) ? u.getColor() : color;
			}
			Node n = new Node(key.get(), edges, weights, cost, color);
			output.collect(key, new Text(n.getLine()));
		}
	}

	/** The main driver for word count map/reduce program. Invoke this method to submit the map/reduce job.
	     @throws IOException When there is communication problems with the job tracker. */
	public int run(String[] args) throws Exception {
		//Get command line arguments. -i <#Iterations>  is required.
//		int maxIters = 11, mapNum = 3, redNum = 3;
		int mapNum = 3, redNum = 3;

		for (int i = 0; i < args.length; ++i) {
			mapNum = ("-m".equals(args[i])) ? Integer.parseInt(args[++i]) : mapNum;
			redNum = ("-r".equals(args[i])) ? Integer.parseInt(args[++i]) : redNum;
//			maxIters = ("-i".equals(args[i])) ? Integer.parseInt(args[++i]) : maxIters;
		}
		if (maxIters < 1) {
			System.err.println("Usage: -i <# of iterations> is a required command line argument");
			System.exit(2);
		}

		for(int iters = 0; iters < maxIters; iters++) {
			println("\t\t\t\t\t\t\t" + "=============" + iters + "===============");
			JobConf conf = getJobConf(args, mp, rd);
			String input = (iters == 0) ? "input" : "output-graph-" + iters;
			FileInputFormat.setInputPaths(conf, new Path(input));
			FileOutputFormat.setOutputPath(conf, new Path("output-graph-" + (iters + 1)));
			JobClient.runJob(conf);
		}
		return 0;
	}
	
	private JobConf getJobConf(String[] args, int mapNum, int redNum) {
		JobConf conf = new JobConf(getConf(), CC6.class);
		conf.setJobName("graphsearch");
		conf.setOutputKeyClass(IntWritable.class);
		conf.setOutputValueClass(Text.class);
		conf.setMapperClass(MapClass.class);
		conf.setReducerClass(Reduce.class);
		conf.setNumMapTasks(mapNum);
		conf.setNumReduceTasks(redNum);
		return conf;
	}
	
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		clearOutput(conf);
		int res = ToolRunner.run(conf, new CC6(), args);
		combineOutputs(conf, "output-graph");
		System.exit(res);
	}
}