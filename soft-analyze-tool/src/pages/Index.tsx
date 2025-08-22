import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Code, FileText, Zap, CheckCircle, AlertTriangle, XCircle, BarChart3, Target } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import axios from "axios";


interface AnalysisResult {
  metrics: {
  loc: number;
  "v(g)": number;
  "ev(g)": number;
  "iv(g)": number;
  d: number;
  lOCode: number;
  uniq_Op: number;
  uniq_Opnd: number;
  total_Op: number;
  branchCount: number;
  };
  prediction: {
    class: number;
    label: string;
    probabilities: {
      "High Quality": number;
      "Low Quality": number;
      "Medium Quality": number;
    };
  };
  problems: string[];
  recommendations: string[];
}

const Index = () => {
  const [code, setCode] = useState("");
  const [softwareName, setSoftwareName] = useState("");
  const [language, setLanguage] = useState<"python" | "javascript" | "">("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const { toast } = useToast();

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  if (!code.trim() || !softwareName.trim() || !language) {
    toast({
      title: "Missing Information",
      description: "Please fill in all fields before submitting.",
      variant: "destructive",
    });
    return;
  }

  setIsAnalyzing(true);

  try {
    const response = await axios.post(
      "http://127.0.0.1:5000/analyze",
      {
        source_code: code,
        software_name: softwareName,
        language: language // 'python' | 'javascript'
      },
      { headers: { "Content-Type": "application/json" } }
    );

    setResults(response.data);

    toast({
      title: "Analysis Complete!",
      description: "Your code has been successfully analyzed.",
    });
  } catch (error: any) {
    toast({
      title: "Analysis Failed",
      description: error.response?.data?.error || "There was an error analyzing your code.",
      variant: "destructive",
    });
  } finally {
    setIsAnalyzing(false);
  }
};


  const getQualityColor = (label: string) => {
    if (label === "High Quality") return "text-emerald-600 dark:text-emerald-400";
    if (label === "Medium Quality") return "text-amber-600 dark:text-amber-400";
    return "text-red-600 dark:text-red-400";
  };

  const getQualityBadgeVariant = (label: string) => {
    if (label === "High Quality") return "default";
    if (label === "Medium Quality") return "secondary";
    return "destructive";
  };

  const getProbabilityColor = (value: number) => {
    if (value >= 0.7) return "text-emerald-600 dark:text-emerald-400";
    if (value >= 0.4) return "text-amber-600 dark:text-amber-400";
    return "text-red-600 dark:text-red-400";
  };

  return (
    <div className="min-h-screen bg-[image:var(--gradient-subtle)]">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Code className="h-8 w-8 text-primary mr-3" />
            <h1 className="text-4xl font-bold bg-[image:var(--gradient-primary)] bg-clip-text text-transparent">
              Devrate
            </h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Intelligent Peer Review and Quality Scoring Platform
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto h-fit">
          {/* Input Form */}
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center">
                <FileText className="h-5 w-5 mr-2" />
                Code Analysis
              </CardTitle>
              <CardDescription>
                Paste your code below and get comprehensive quality insights
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Software Name */}
                <div className="space-y-2">
                  <Label htmlFor="software-name">Software/Project Name</Label>
                  <Input
                    id="software-name"
                    placeholder="Enter your project name..."
                    value={softwareName}
                    onChange={(e) => setSoftwareName(e.target.value)}
                    className="transition-all focus:ring-2 focus:ring-primary/20"
                  />
                </div>

                {/* Language */}
                <div className="space-y-2">
                  <Label htmlFor="language">Programming Language</Label>
                  <Select value={language} onValueChange={(value: "python" | "javascript") => setLanguage(value)}>
                    <SelectTrigger className="transition-all focus:ring-2 focus:ring-primary/20">
                      <SelectValue placeholder="Select language..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="python">Python</SelectItem>
                      <SelectItem value="javascript">JavaScript</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Code Input */}
                <div className="space-y-2">
                  <Label htmlFor="code">Code</Label>
                  <Textarea
                    id="code"
                    placeholder="Paste your code here..."
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    className="min-h-[300px] font-mono text-sm transition-all focus:ring-2 focus:ring-primary/20"
                  />
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-[image:var(--gradient-primary)] hover:opacity-90 transition-opacity shadow-[var(--shadow-elegant)]" 
                  disabled={isAnalyzing}
                  size="lg"
                >
                  {isAnalyzing ? (
                    <>
                      <LoadingSpinner size="sm" className="mr-2" />
                      Analyzing Code...
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      Analyze Code Quality
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Results Display */}
          <div className="space-y-6">
            {results ? (
              <>
                {/* Quality Prediction */}
                <Card className="shadow-[var(--shadow-elegant)] border-0 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center text-lg">
                      <Target className="h-5 w-5 mr-2" />
                      Quality Prediction
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center">
                      <div className={`text-5xl font-bold mb-2 ${getQualityColor(results.prediction.label)}`}>
                        {(results.prediction.probabilities[results.prediction.label] * 100).toFixed(0)}%
                      </div>
                      <Badge variant={getQualityBadgeVariant(results.prediction.label)} className="text-sm px-3 py-1 mb-4">
                        {results.prediction.label}
                      </Badge>
                      <div className="grid grid-cols-3 gap-2 mt-4">
                        {Object.entries(results.prediction.probabilities).map(([quality, probability]) => (
                          <div key={quality} className="text-center p-2 rounded-lg bg-muted/30">
                            <div className={`text-sm font-bold ${getProbabilityColor(probability)}`}>
                              {(probability * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {quality.replace(" Quality", "")}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Code Metrics */}
                <Card className="shadow-[var(--shadow-elegant)] border-0 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center text-lg">
                      <BarChart3 className="h-5 w-5 mr-2" />
                      Code Metrics
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(results.metrics).map(([key, value]) => (
                        <div key={key} className="text-center p-3 rounded-lg bg-gradient-to-br from-muted/50 to-muted/20">
                          <div className="text-2xl font-bold text-primary">
                            {typeof value === 'number' ? Math.round(value) : 'N/A'}
                          </div>
                          <div className="text-sm text-muted-foreground font-mono">
                            {key}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

              {/* Problems */}
                {results.problems.length > 0 && (
                  <Card className="shadow-[var(--shadow-elegant)] border-0 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center text-lg">
                        <AlertTriangle className="h-5 w-5 mr-2 text-amber-500" />
                        Problems Detected
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {results.problems.map((problem, index) => (
                          <div key={index} className="flex items-start space-x-3 p-3 rounded-lg bg-gradient-to-r from-amber-50/50 to-orange-50/50 dark:from-amber-950/20 dark:to-orange-950/20 border border-amber-200/50 dark:border-amber-800/50">
                            <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm">{problem}</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Recommendations */}
                <Card className="shadow-[var(--shadow-elegant)] border-0 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center text-lg">
                      <CheckCircle className="h-5 w-5 mr-2 text-emerald-500" />
                      Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-3">
                      {results.recommendations.map((recommendation, index) => (
                        <li key={index} className="flex items-start space-x-3 p-3 rounded-lg bg-gradient-to-r from-emerald-50/50 to-teal-50/50 dark:from-emerald-950/20 dark:to-teal-950/20 border border-emerald-200/50 dark:border-emerald-800/50">
                          <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{recommendation}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>              </>
            ) : (
              <Card className="shadow-lg">
                <CardContent className="pt-6">
                  <div className="text-center py-12">
                    <Code className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-medium mb-2">Ready to Analyze</h3>
                    <p className="text-muted-foreground">
                      Fill out the form and submit your code to see quality analysis results here.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
