import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

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
			Node composite = new Node(key.get());
			print(key);
			if(vals.size() == 1) {
				println("\t" + vals.get(0));
				composite = new Node(key.get() + "\t" + vals.get(0));	
			}
			
			else {
				int cost = Integer.MAX_VALUE;
				Color color = null;

				List<Node> nodes = vals.stream().map(x->new Node(key.get() + "\t" + x)).collect(Collectors.toList());
				int minCostIndex = 0;
				//GET EDGES AND WEIGHTS
				for(int i  = 0; i < nodes.size(); i++) {
					println("\t" + vals.get(i));
					Node n = nodes.get(i);
					if(n.getEdges().size() > 0) {
						composite.setEdges(n.getEdges());
						composite.setWeights(n.getWeights());
					}
					if(n.getCost() < cost) {
						cost = n.getCost();
						minCostIndex = i;
					}
					if(n.getColor() == Color.WHITE)
						color = Color.GRAY;
				}
				composite.setColor(color == null ? nodes.get(minCostIndex).getColor() : color);
				composite.setCost(cost);
			}
			output.collect(key, new Text(composite.getLine()));
		}
	}

	/** The main driver for word count map/reduce program. Invoke this method to submit the map/reduce job.
	     @throws IOException When there is communication problems with the job tracker. */
	public int run(String[] args) throws Exception {
		//Get command line arguments. -i <#Iterations>  is required.
		int maxIters = 11, mapNum = 3, redNum = 3;

		for (int i = 0; i < args.length; ++i) {
			mapNum = ("-m".equals(args[i])) ? Integer.parseInt(args[++i]) : mapNum;
			redNum = ("-r".equals(args[i])) ? Integer.parseInt(args[++i]) : redNum;
			maxIters = ("-i".equals(args[i])) ? Integer.parseInt(args[++i]) : maxIters;
		}
		if (maxIters < 1) {
			System.err.println("Usage: -i <# of iterations> is a required command line argument");
			System.exit(2);
		}

		for(int iters = 0; iters < maxIters; iters++) {
			println("=========" + iters + "==========");
			JobConf conf = getJobConf(args, mapNum, redNum);
			String input = (iters == 0) ? "input" : "output-graph-" + iters;
			FileInputFormat.setInputPaths(conf, new Path(input));
			FileOutputFormat.setOutputPath(conf, new Path("output-graph-" + (iters + 1)));
			JobClient.runJob(conf);
			println("");
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