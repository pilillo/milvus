package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

var (
	collectionName = "my_collection"
	milvusClient   client.Client
)

func connect() {
	var err error
	if milvusClient, err = client.NewGrpcClient(
		context.Background(), // ctx
		"localhost:19530",    // addr
	); err != nil {
		log.Fatal("failed to create milvus client:", err.Error())
	}
}

func createCollection() {
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "Test book search",
		Fields: []*entity.Field{
			{
				Name:       "book_id",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:       "word_count",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: false,
				AutoID:     false,
			},
			{
				Name:     "book_intro",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": "2",
				},
			},
		},
	}

	exists, err := milvusClient.HasCollection(context.Background(), collectionName)
	if err != nil {
		log.Fatal("failed to check collection existence:", err.Error())
	}
	if !exists {
		// Create a new collection
		err := milvusClient.CreateCollection(
			context.Background(), // ctx
			schema,
			2, // shardNum
		)
		if err != nil {
			log.Fatal("failed to create collection:", err.Error())
		}
	}
}

func insertEntry(columns ...entity.Column) {
	if _, err := milvusClient.Insert(
		context.Background(), // ctx
		collectionName,
		"", // partitionName
		columns...,
	); err != nil {
		log.Fatal("failed to insert data:", err.Error())
	}
}

var index *entity.IndexIvfFlat

func createIndex() {
	var err error
	/*entity.NewIndexHNSW(
	entity.IP, // metricType
	1024,      // ConstructParams
	*/
	if index, err = entity.NewIndexIvfFlat(
		entity.L2,
		1024); err != nil {
		log.Fatal("failed to create index:", err.Error())
	}

	if err := milvusClient.CreateIndex(
		context.Background(), // ctx
		collectionName,       // collName
		"book_intro",         // fieldName
		index,
		false, // async
	); err != nil {
		log.Fatal("failed to create index:", err.Error())
	}
}

func createPartition(partitionName string, shardNum int) {
	if err := milvusClient.CreatePartition(
		context.Background(),
		collectionName,
		partitionName,
	); err != nil {
		log.Fatal("failed to create partition:", err.Error())
	}
}

func loadCollection() {
	if err := milvusClient.LoadCollection(
		context.Background(),
		collectionName,
		false, // async
	); err != nil {
		log.Fatal("failed to load collection:", err.Error())
	}
}

func search() ([]client.SearchResult, error) {
	sp, _ := entity.NewIndexFlatSearchParam( // NewIndex*SearchParam func
		10, // searchParam
	)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(timeout))
	defer cancel()
	return milvusClient.Search(
		ctx,                 // ctx
		collectionName,      // CollectionName
		[]string{},          // partitionNames
		"",                  // expr
		[]string{"book_id"}, // outputFields
		[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})}, // vectors
		"book_intro", // vectorField
		entity.L2,    // metricType
		2,            // topK
		sp,           // sp
	)
}

var timeout = time.Second * 10

func hybridSearch() ([]client.SearchResult, error) {
	sp, _ := entity.NewIndexFlatSearchParam( // NewIndex*SearchParam func
		10, // searchParam
	)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(timeout))
	defer cancel()
	return milvusClient.Search(
		ctx,                   // ctx
		collectionName,        // CollectionName
		[]string{},            // partitionNames
		"word_count <= 11000", // expr
		[]string{"book_id"},   // outputFields
		[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})}, // vectors
		"book_intro", // vectorField
		entity.L2,    // metricType
		2,            // topK
		sp,           // sp
	)
}

func main() {
	connect()
	defer milvusClient.Close()

	createCollection()
	createIndex()

	bookIDs := make([]int64, 0, 2000)
	wordCounts := make([]int64, 0, 2000)
	bookIntros := make([][]float32, 0, 2000)
	for i := 0; i < 2000; i++ {
		bookIDs = append(bookIDs, int64(i))
		wordCounts = append(wordCounts, int64(i+10000))
		v := make([]float32, 0, 2)
		for j := 0; j < 2; j++ {
			v = append(v, rand.Float32())
		}
		bookIntros = append(bookIntros, v)
	}
	idColumn := entity.NewColumnInt64("book_id", bookIDs)
	wordColumn := entity.NewColumnInt64("word_count", wordCounts)
	introColumn := entity.NewColumnFloatVector("book_intro", 2, bookIntros)

	insertEntry(idColumn, wordColumn, introColumn)
	loadCollection()

	searchResult, err := search()
	//searchResult, err := hybridSearch()
	if err != nil {
		log.Fatal("fail to search collection:", err.Error())
	}

	fmt.Printf("%#v\n", searchResult)
	for _, sr := range searchResult {
		fmt.Println(sr.IDs)
		fmt.Println(sr.Scores)
	}

}
