import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.hadoop.io.Text;

public class Node {
	public static enum Color { WHITE, GRAY, BLACK };
	private int id, cost;
	private List<Integer> edges = new ArrayList<>(), weights = new ArrayList<>();
	private Color color = Color.WHITE;

	public Node(String str) {
		String[] map = str.split("\t");
		String key = map[0], value = map[1];
		String[] tokens = value.split("\\|");
		
		//!!!!!!!!!!!! FIX WEIGHTS WORKAROUND
		if(tokens.length == 3)
			tokens = new String[] {tokens[0], tokens[0], tokens[1], tokens[2]};

		//ID
		this.id = Integer.parseInt(key);
		//EDGES
		Arrays.stream(tokens[0].split(",")).filter(x->x.length()>0).forEach(x->edges.add(Integer.parseInt(x)));
		//WEIGHTS
		Arrays.stream(tokens[1].split(",")).filter(x->x.length()>0).forEach(x->weights.add(Integer.parseInt(x)));
		//COST
		this.cost = (tokens[2].equals("Integer.MAX_VALUE")) ? Integer.MAX_VALUE : Integer.parseInt(tokens[2]);
		//COLOR
		this.color = Color.valueOf(tokens[3]);
	}

	//!!!!!! FIX WEIGHTS WORKAROUND
	public Node(int id) { this.id = id; weights.add(0); }
	public int getId() { return this.id; }

	public List<Integer> getEdges() { return this.edges; }
	public void setEdges(List<Integer> edges) { this.edges = edges; }

	public List<Integer> getWeights() { return this.weights; }
	public void setWeights(List<Integer> weights) { this.weights = weights; }
	
	public int getCost() { return this.cost; }
	public void setCost(int cost) { this.cost = cost; }

	public Color getColor() { return this.color; }
	public void setColor(Color color) { this.color = color; }

	public Text getLine() {
		StringBuffer s = new StringBuffer();
		s.append(edges.stream().map(x->x.toString()).collect(Collectors.joining(","))).append("|");
		s.append(weights.stream().map(x->x.toString()).collect(Collectors.joining(","))).append("|");
		s.append(this.cost < Integer.MAX_VALUE ? this.cost : "Integer.MAX_VALUE").append("|");
		s.append(color.toString());
		return new Text(s.toString());
	}
}