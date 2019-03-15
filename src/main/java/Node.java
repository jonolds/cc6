import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.hadoop.io.Text;

public class Node {
	public static enum Color { WHITE, GRAY, BLACK };
	private int id, distance;
	private List<Integer> edges = new ArrayList<Integer>();
	private Color color = Color.WHITE;

	public Node(String str) {
		String[] map = str.split("\t");
		String key = map[0], value = map[1];
		String[] tokens = value.split("\\|");

		//ID
		this.id = Integer.parseInt(key);
		//EDGES
		Arrays.stream(tokens[0].split(",")).filter(x->x.length()>0).forEach(x->edges.add(Integer.parseInt(x)));
		
		//!!!!!!!!!!!!FIX
		System.out.println(tokens.length);
		if(tokens.length == 4) {
			tokens[1] = tokens[2];
			tokens[2] = tokens[3];
		}
			
		
		//DISTANCE
		this.distance = (tokens[1].equals("Integer.MAX_VALUE")) ? Integer.MAX_VALUE : Integer.parseInt(tokens[1]);
		//COLOR
		this.color = Color.valueOf(tokens[2]);
	}

	public Node(int id) { this.id = id; }
	public int getId() { return this.id; }

	public List<Integer> getEdges() { return this.edges; }
	public void setEdges(List<Integer> edges) { this.edges = edges; }
	
	public int getDistance() { return this.distance; }
	public void setDistance(int distance) { this.distance = distance; }

	public Color getColor() { return this.color; }
	public void setColor(Color color) { this.color = color; }

	public Text getLine() {
		StringBuffer s = new StringBuffer();
		s.append(edges.stream().map(x->x.toString()).collect(Collectors.joining(","))).append("|");
		s.append(this.distance < Integer.MAX_VALUE ? this.distance : "Integer.MAX_VALUE").append("|");
		s.append(color.toString());
		return new Text(s.toString());
	}
}