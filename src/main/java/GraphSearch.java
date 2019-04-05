import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Spliterators;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

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
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class GraphSearch extends CC6Helper implements Tool {
	/** WHITE and BLACK nodes are emitted as is. For every edge of a GRAY node, we emit a new Node with 
	 * distance incremented by one. The Color.GRAY node is then colored black and is also emitted. */
	public static class MapClass extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
		
		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
			Node node = new Node(value.toString());
			if (node.getColor() == Color.GRAY) {
				for(int i = 0; i < node.getEdges().size(); i++)
					output.collect(new IntWritable(node.getEdges().get(i)), node.getSpawnText(i));
				node.setColor(Color.BLACK);
			}
			// Emit current node. White and Black are sent as is. Grays have been set to Black in previous block.
			output.collect(new IntWritable(node.getId()), node.getLine());
		}
	}

	/** A reducer class that just emits the sum of the input values. */
	public static class Reduce extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
		/** Make a new node which combines all information for this single node id. The Node should have
		 * - 1)The full list of edges. 2)The minimum distance. 3)The darkest Color. */
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
			List<String> vals = iterFiniteStream(values).map(x->x.toString()).collect(Collectors.toList());
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
		int maxIters = 0, mapNum = 3, redNum = 3;

		for (int i = 0; i < args.length; ++i) {
			mapNum = ("-m".equals(args[i])) ? Integer.parseInt(args[++i]) : mapNum;
			redNum = ("-r".equals(args[i])) ? Integer.parseInt(args[++i]) : redNum;
			maxIters = ("-i".equals(args[i])) ? Integer.parseInt(args[++i]) : maxIters;
		}
		if (maxIters < 1) {
			System.err.println("Usage: -i <# of iterations> is a required command line argument");
			System.exit(2);
		}
		int failCount = 0;
		for(int iters = 0; iters < maxIters; iters++) {
			println("=========" + iters + "==========");
			JobConf conf = getJobConf(args, mapNum, redNum);
			String input = (iters == 0) ? "input-graph" : "output" + File.separator + "output-graph-" + iters;
			FileInputFormat.setInputPaths(conf, new Path(input));
			FileOutputFormat.setOutputPath(conf, new Path("output" + File.separator + "output-graph-" + (iters + 1)));
			RunningJob job = JobClient.runJob(conf);
			failCount += job.isSuccessful() ? 0 : 1;
			println("");
		}
		return failCount;
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
		return conf;
	}
	
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		clearOutput(conf);
		int res = ToolRunner.run(conf, new GraphSearch(), args);
		println(res == 0 ? "SUCCESS!!!!!" : "FAILURE!!!!");
		combineOutputs(conf, "output-graph");
		System.exit(res);
	}
}

class CC6Helper extends Configured {
	
	static void clearOutput(Configuration conf) throws IllegalArgumentException, IOException {
		List<String> outDirs = Files.list(Paths.get("")).map(x->x.toString()).filter(x->x.startsWith("output")).collect(Collectors.toList());
		for(String folder : outDirs)
			new Path(folder).getFileSystem(conf).delete(new Path(folder), true);
	}
	
	static void combineOutputs(Configuration conf, String outDirPrefix) throws IOException {
		List<String> outputStringsList = new ArrayList<>();
		List<String> outDirs = getFilesStartingWithInDir(outDirPrefix, "output");
		Collections.sort(outDirs);
		for(String outFolder : outDirs)
			outputStringsList.add(getCombinedOutputsInFolderAsString(conf, outFolder));
		String output = outputStringsList.stream().collect(Collectors.joining("\n"));
		BufferedWriter bf = new BufferedWriter(new FileWriter("output" + File.separator + "outAll.txt"));
		bf.write(output);
		bf.close();
	}

	static String getCombinedOutputsInFolderAsString(Configuration conf, String outFolder) throws IOException {
		List<File> out_parts = getFilesStartingWithInDir("part", outFolder).stream().map(x->new File(x)).collect(Collectors.toList());
		List<String> out_lines = new ArrayList<>();
		for(File file: out_parts)
			out_lines.addAll(Files.readAllLines(file.toPath()));
		Collections.sort(out_lines);
		return outFolder + "\n\t" + out_lines.stream().collect(Collectors.joining("\n\t"));
	}
	
	static List<String> getFilesStartingWithInDir(String start, String dir) throws IOException {
		List<String> dirs = new ArrayList<>();
		Files.walk(Paths.get(dir), 1).filter(x->x != new Path(dir) && x.toString().startsWith(dir + File.separator + start)).forEach(x->dirs.add(x.toString()));
		return dirs;
	}
	
	static <T> Stream<T> iterFiniteStream(final Iterator<T> iterator) {
	    return StreamSupport.stream(Spliterators.spliteratorUnknownSize(iterator, 0), false);
	}
	
	static <T>void println(T t) { System.out.println(t.toString()); }
	static <T>void print(T t) { System.out.print(t.toString()); }
}