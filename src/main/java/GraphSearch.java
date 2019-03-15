import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
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

/** Input is a map in adjacency list format, and performs a breadth-first search. The input format is:
 * "ID	 EDGES|DISTANCE|COLOR"
 * 		ID = the unique identifier for a node (assumed to be an int here)
 * 		EDGES = the list of edges emanating from the node (e.g. 3,8,9,12)
 * 		DISTANCE = the to be determined distance of the node from the source
 * 		COLOR = a simple status tracking field to keep track of when we're finished with a node
 * Source should have distance 0 and be GRAY. Others have distance Integer.MAX_VALUE and color WHITE.*/
public class GraphSearch extends Configured implements Tool {
	public static final Log LOG = LogFactory.getLog("org.apache.hadoop.examples.GraphSearch");
	
	/** Nodes that are Color.WHITE or Color.BLACK are emitted, as is. For every
	 * edge of a Color.GRAY node, we emit a new Node with distance incremented by
	 * one. The Color.GRAY node is then colored black and is also emitted. */
	public static class MapClass extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
//			LOG.info("MAP EXECUTING FOR KEY [" + key.toString() + "] and value [" + value.toString() + "]");
			println("M " + value);
			Node node = new Node(value.toString());

			// For each GRAY node, emit each of the edges as a new node (also GRAY)
			if (node.getColor() == Node.Color.GRAY) {
//				for (int edge : node.getEdges()) {
				for(int i = 0; i < node.getEdges().size(); i++) {
					Node vnode = new Node(node.getEdges().get(i));
					vnode.setCost(node.getCost() + node.getWeights().get(i));
					vnode.setColor(Node.Color.GRAY);
					output.collect(new IntWritable(vnode.getId()), vnode.getLine());
				}
				node.setColor(Node.Color.BLACK);
			}
			output.collect(new IntWritable(node.getId()), node.getLine());
//			LOG.info("MAP OUTPUTTING FOR KEY [" + node.getId() + "] and value [" + node.getLine() + "]");
		}
	}

	/** A reducer class that just emits the sum of the input values. */
	public static class Reduce extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {

		/** Make a new node which combines all information for this single node id. The Node should have
		 * - 1)The full list of edges. 2)The minimum distance. 3)The darkest Color. */
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
//			LOG.info("REDUCE EXECUTING FOR INPUT KEY [" + key.toString() + "]");
			
			List<String> vals = new ArrayList<>();
			while(values.hasNext())
				vals.add(values.next().toString());
			
			List<Integer> edges = null;
			List<Integer> weights = null;
			int cost = Integer.MAX_VALUE;
			Node.Color color = Node.Color.WHITE;

			for(String value : vals) {
				println("R " + key + "\t" + value);
//				Text value = values.next();
				Node u = new Node(key.get() + "\t" + value);

				// One one copy of the node will be the fully expanded version, which includes the edges
				if(u.getEdges().size() > 0)
					edges = u.getEdges();
				if(u.getWeights().size() > 0)
					weights = u.getWeights();
					
				// Save the minimum cost
				if(u.getCost() < cost)
					cost = u.getCost();
				// Save the darkest color
				if(u.getColor().ordinal() > color.ordinal())
					color = u.getColor();
			}

			Node n = new Node(key.get());
			n.setCost(cost);
			n.setEdges(edges);
			n.setWeights(weights);
			n.setColor(color);
			output.collect(key, new Text(n.getLine()));
//			LOG.info("REDUCE OUTPUTTING FOR FINAL KEY [" + key + "] and value [" + n.getLine() + "]");
		}
	}

	/** The main driver for word count map/reduce program. Invoke this method to submit the map/reduce job.
	     @throws IOException When there is communication problems with the job tracker. */
	public int run(String[] args) throws Exception {
		//Get command line arguments. -i <#Iterations>  is required.
		int maxIters = 4, mapNum = 3, redNum = 3;
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
			JobConf conf = getJobConf(args, mapNum, redNum);
			String input = (iters == 0) ? "input" : "output-graph-" + iters;
			FileInputFormat.setInputPaths(conf, new Path(input));
			FileOutputFormat.setOutputPath(conf, new Path("output-graph-" + (iters + 1)));
			JobClient.runJob(conf);
		}
		return 0;
	}
	
	private JobConf getJobConf(String[] args, int mapNum, int redNum) {
		JobConf conf = new JobConf(getConf(), GraphSearch.class);
		conf.setJobName("graphsearch");
		conf.setOutputKeyClass(IntWritable.class);
		conf.setOutputValueClass(Text.class);
		conf.setMapperClass(MapClass.class);
		conf.setReducerClass(Reduce.class);
		conf.setNumMapTasks(mapNum);
		conf.setNumReduceTasks(redNum);

		LOG.info("The number of reduce tasks has been set to " + conf.getNumReduceTasks());
		LOG.info("The number of mapper tasks has been set to " + conf.getNumMapTasks());
		return conf;
	}
	
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		deleteOutputFolders(conf);
		int res = ToolRunner.run(conf, new GraphSearch(), args);
		combineOutputFolders(conf);
		System.exit(res);
	}
	
	static int printUsage() {
		System.out.println("graphsearch [-m <num mappers>] [-r <num reducers>]");
		ToolRunner.printGenericCommandUsage(System.out);
		return -1;
	}
	
	public static void deleteOutputFolders(Configuration conf) throws IllegalArgumentException, IOException {
		List<String> folders = Files.list(Paths.get("")).map(x->x.toString()).filter(x->x.startsWith("output")).collect(Collectors.toList());
		for(String folder : folders)
			new Path(folder).getFileSystem(conf).delete(new Path(folder), true);
	}	
	public static void combineOutputFolders(Configuration conf) throws IOException {
		List<String> outDirs = Files.list(Paths.get("")).map(x->x.toString()).filter(x->x.startsWith("output-graph")).collect(Collectors.toList());
		new File("output").mkdir();
		for(String outDir : outDirs) {
			List<String> files = Files.list(Paths.get(outDir + "/")).map(x->x.toString()).filter(x->!(x.endsWith("crc")||x.endsWith("SUCCESS"))).collect(Collectors.toList());
			for(String file : files) {
				String newName = file.substring(13).replace('\\', '-');
				Files.copy(new File(file).toPath(), new File("output\\" + newName + ".txt").toPath());
			}
			new Path(outDir).getFileSystem(conf).delete(new Path(outDir), true);
		}
	}
	public static <T>void println(T t) { System.out.println(t.toString()); }
}