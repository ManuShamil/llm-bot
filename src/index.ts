import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { ChatOpenAI } from 'langchain/chat_models/openai'

import { RetrievalQAChain } from 'langchain/chains'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'

import { Document } from 'langchain/dist/document'
import fs from 'fs'

const URLS_FILE = 'urls.txt'
const DOC_URLS = fs.readFileSync( URLS_FILE, 'utf8').split(`\r\n`)

class KBBot {
    private embeddings: OpenAIEmbeddings
    private memoryStore: MemoryVectorStore
    private retrievalChain: RetrievalQAChain

    public constructor() {
        this.embeddings = new OpenAIEmbeddings()    
        this.memoryStore = new MemoryVectorStore( this.embeddings )
        this.retrievalChain = RetrievalQAChain.fromLLM( new ChatOpenAI(), this.memoryStore.asRetriever() )
    }

    private loadDocument<T>( url: string ) {
        const urlSplit = url.split('/')
        const fileName = `data/${urlSplit[urlSplit.length - 1]}.json`
        if ( !fs.existsSync( fileName ) )
            return undefined
        return JSON.parse( fs.readFileSync( fileName, 'utf8' ) ) as T
    }

    private saveDocument<T>( url: string, data: T  ) {
        const urlSplit = url.split('/')
        const fileName = `data/${urlSplit[urlSplit.length - 1]}.json`
        fs.writeFileSync( fileName, JSON.stringify( data ) )
    }

    public async addWebDocument( url: string ) {

        console.info( `Adding document from ${url}` )

        let chunks = this.loadDocument<Document<Record<string, any>>[]>( url )
        if ( chunks ) {
            console.info( `Document already exists, skipping` )
            await this.memoryStore.addDocuments( chunks )
            return
        }

        let data = await new CheerioWebBaseLoader( url ).load()
        let textSplitter = new RecursiveCharacterTextSplitter( { chunkSize: 500, chunkOverlap: 0 } )
        chunks = await textSplitter.splitDocuments( data )
        this.saveDocument( url, chunks )
        await this.memoryStore.addDocuments( chunks )
    }
    
    public retrieve(query: string, callback: (text: string, query: string) => void) {
        this.retrievalChain.call({ query }).then((result) => {
            const { text } = result;
            callback( text, query );
        })
      }

    public async retrieveSync( query: string ) {
        return await this.retrievalChain.call( { query } )
    }

    public async build( urls: string[] ) {
        for( let url of urls )
            await this.addWebDocument( url )
    }
}

const qaBot = new KBBot()

import * as readline from 'readline'

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
})


function getUserInput() {
    rl.question('Enter your query: ', async (userInput: string) => {
        if (userInput.toLowerCase() == 'exit')
            rl.close()
        else {
            try {
                let { text } = await qaBot.retrieveSync( userInput )
                console.log( text )
            } catch( e ) {
                console.log( e )
            }

            getUserInput()
        }
    })
  }

async function main() {
    await qaBot.build( DOC_URLS )
    getUserInput();

    rl.on('close', () => process.exit(0));
}

main()
